import pathlib
import cv2
import random
import numpy as np
import scipy.ndimage as ndimage
import sklearn as sk
from datetime import datetime
from PIL import Image
from experiment_log import log_generation


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "oem_sar"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_labels(path):
    return np.asarray(Image.open(path)).astype(np.uint8)


def color_model_output(result_root):
    from inference import color_model_output as _color_model_output

    return _color_model_output(result_root=result_root)


def generated_directory_name(noise_method, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now()
    if isinstance(timestamp, str):
        timestamp_text = timestamp
    else:
        timestamp_text = timestamp.strftime("%y%m%d_%H%M%S")
    return f"{noise_method}_{timestamp_text}"


def creation_tstamp(timestamp=None):
    if timestamp is None:
        timestamp = datetime.now()
    if timestamp is None:
        timestamp = datetime.now()
    if isinstance(timestamp, str):
        return timestamp
    return timestamp.strftime("%y%m%d_%H%M%S")


def add_random_pixel_noise(y_true, n_classes, noise_rate, class_ids=None):
    """
    Add random label noise at pixel level.

    The number of noisy pixels sampled from each class is proportional to that
    class's label count in the full test dataset. By default, background class 0
    is left unchanged.
    """
    y_true = np.asarray(y_true)

    if class_ids is None:
        class_ids = np.arange(1, n_classes)
    class_ids = np.asarray(class_ids, dtype=np.int64)

    noisy = y_true.copy()
    replacement_choices = np.arange(n_classes)

    for class_id in class_ids:
        pixel_locations = np.argwhere(y_true == class_id)
        n_class_pixels = len(pixel_locations)
        if n_class_pixels == 0:
            continue

        n_noisy_pixels = int(round(noise_rate * n_class_pixels))
        if n_noisy_pixels == 0:
            continue

        selected_idx = np.random.choice(n_class_pixels, size=n_noisy_pixels, replace=False)
        selected_locations = pixel_locations[selected_idx]
        valid_replacements = replacement_choices[replacement_choices != class_id]
        new_labels = np.random.choice(valid_replacements, size=n_noisy_pixels)
        noisy[tuple(selected_locations.T)] = new_labels

    return noisy


def boundary_dilation_erosion(y_true, class_ids, mean, fallback_class=0):
    """
    Randomly dilate or erode connected objects of selected classes.

    For each object, a magnitude is sampled from N(mean, 0.1 * mean), clipped
    at zero, and rounded to a whole number of pixels. An independent sign then
    selects dilation or erosion with equal probability. Eroded pixels are
    filled pixelwise from the nearest exterior label produced by preceding
    erosions. All erosions are applied before any dilations.
    """
    y_true = np.asarray(y_true)
    if y_true.ndim != 3:
        raise ValueError("'y_true' must have shape (images, height, width).")
    if not np.isfinite(mean) or mean < 0:
        raise ValueError("'mean' must be a finite, non-negative number.")
    if isinstance(class_ids, (int, np.integer)):
        class_ids = [int(class_ids)]
    class_ids = [int(class_id) for class_id in class_ids]

    noisy = y_true.copy()
    structure = np.ones((3, 3), dtype=bool)
    magnitude_std = 0.1 * mean

    for i, mask in enumerate(y_true):
        noisy_mask = noisy[i]
        h, w = mask.shape
        plans = []

        for class_id in class_ids:
            class_mask = mask == class_id
            if not class_mask.any():
                continue

            labeled, n_objects = ndimage.label(class_mask, structure=structure)
            object_slices = ndimage.find_objects(labeled)
            magnitudes = np.maximum(
                np.random.normal(
                    loc=mean, scale=magnitude_std, size=n_objects
                ),
                0.0,
            )
            signs = 2 * np.random.binomial(1, 0.5, size=n_objects) - 1
            amounts = (signs * np.rint(magnitudes)).astype(np.int32)
            plans.append((class_id, labeled, object_slices, amounts))

        for class_id, labeled, object_slices, amounts in plans:
            for obj_id, (obj_slice, amount) in enumerate(
                zip(object_slices, amounts), start=1
            ):
                if obj_slice is None or amount >= 0:
                    continue

                iterations = abs(int(amount))
                y_slice, x_slice = obj_slice
                pad = iterations + 1
                y0 = max(y_slice.start - pad, 0)
                y1 = min(y_slice.stop + pad, h)
                x0 = max(x_slice.start - pad, 0)
                x1 = min(x_slice.stop + pad, w)
                local_slice = (slice(y0, y1), slice(x0, x1))

                local_object = labeled[local_slice] == obj_id
                local_noisy = noisy_mask[local_slice].copy()

                eroded = ndimage.binary_erosion(
                    local_object,
                    structure=structure,
                    iterations=iterations,
                    border_value=0,
                )
                changed_region = local_object & ~eroded
                if not changed_region.any():
                    continue

                if (~local_object).any():
                    nearest = ndimage.distance_transform_edt(
                        local_object,
                        return_distances=False,
                        return_indices=True,
                    )
                    local_noisy[changed_region] = local_noisy[
                        nearest[0][changed_region],
                        nearest[1][changed_region],
                    ]
                else:
                    local_noisy[changed_region] = fallback_class
                noisy_mask[local_slice] = local_noisy

        for class_id, labeled, object_slices, amounts in plans:
            for obj_id, (obj_slice, amount) in enumerate(
                zip(object_slices, amounts), start=1
            ):
                if obj_slice is None or amount <= 0:
                    continue

                iterations = int(amount)
                y_slice, x_slice = obj_slice
                pad = iterations + 1
                y0 = max(y_slice.start - pad, 0)
                y1 = min(y_slice.stop + pad, h)
                x0 = max(x_slice.start - pad, 0)
                x1 = min(x_slice.stop + pad, w)
                local_slice = (slice(y0, y1), slice(x0, x1))

                local_object = labeled[local_slice] == obj_id
                changed_region = ndimage.binary_dilation(
                    local_object,
                    structure=structure,
                    iterations=iterations,
                )
                local_noisy = noisy_mask[local_slice].copy()
                local_noisy[changed_region] = class_id
                noisy_mask[local_slice] = local_noisy

    return noisy


def add_semantic_swapping(y_true, y_pred, n_classes, swap_prob, class_ids=None):
    """
    Replace foreground connected components using class-confusion probabilities.

    ``swap_prob`` controls whether an eligible object changes class. When an
    object is selected, the replacement is sampled from the source class's
    confusion row with the source class excluded.
    """
    if not 0.0 <= swap_prob <= 1.0:
        raise ValueError("'swap_prob' must be between 0 and 1.")

    if class_ids is None:
        class_ids = np.arange(1, n_classes)
    if isinstance(class_ids, (int, np.integer)):
        class_ids = [int(class_ids)]
    class_ids = [int(class_id) for class_id in class_ids]
    invalid_class_ids = [
        class_id for class_id in class_ids if class_id < 0 or class_id >= n_classes
    ]
    if invalid_class_ids:
        raise ValueError(
            f"'class_ids' must be between 0 and {n_classes - 1}; "
            f"got {invalid_class_ids}."
        )

    structure = np.ones((3, 3), dtype=np.int32)
    class_choices = np.arange(n_classes)
    confusion_mat = sk.metrics.confusion_matrix(
        y_true.ravel(), y_pred.ravel(), labels=class_choices
    )
    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    relative_confusion_matrix = np.divide(
        confusion_mat,
        row_sums,
        out=np.eye(n_classes, dtype=float),
        where=row_sums != 0,
    )
    replacement_probs_by_class = {}
    for class_id in class_ids:
        replacement_probs = relative_confusion_matrix[class_id].copy()
        replacement_probs[class_id] = 0.0
        replacement_sum = replacement_probs.sum()
        if replacement_sum == 0.0:
            replacement_probs_by_class[class_id] = None
        else:
            replacement_probs_by_class[class_id] = replacement_probs / replacement_sum

    noisy_set = np.empty_like(y_true)

    for i, mask in enumerate(y_true):
        noisy_mask = mask.copy()
        valid_labels = mask[(mask >= 0) & (mask < n_classes)]
        present_classes = np.bincount(
            valid_labels.ravel(), minlength=n_classes
        ).nonzero()[0]

        for class_id in present_classes:
            if class_id not in replacement_probs_by_class:
                continue

            replacement_probs = replacement_probs_by_class[class_id]
            if replacement_probs is None:
                continue

            class_mask = mask == class_id
            labeled, n_objects = ndimage.label(class_mask, structure=structure)
            if n_objects == 0:
                continue

            swap_objects = np.random.rand(n_objects) < swap_prob
            if not swap_objects.any():
                continue

            replacement_classes = np.random.choice(
                class_choices,
                size=int(swap_objects.sum()),
                p=replacement_probs,
            )
            replacement_idx = 0

            for obj_id, object_slice in enumerate(ndimage.find_objects(labeled), start=1):
                if object_slice is None or not swap_objects[obj_id - 1]:
                    continue

                new_class = replacement_classes[replacement_idx]
                replacement_idx += 1
                object_mask = labeled[object_slice] == obj_id
                noisy_mask[object_slice][object_mask] = new_class

        noisy_set[i] = noisy_mask

    return noisy_set


def _sample_shift_vectors(num_images, mean):
    """Sample one integer (dy, dx) shift vector per image."""
    if not np.isfinite(mean) or mean < 0:
        raise ValueError("'mean' must be a finite, non-negative number.")

    std = 0.1 * mean
    distances = np.maximum(
        np.random.normal(loc=mean, scale=std, size=num_images), 0.0
    )
    angles = np.random.uniform(0.0, 2.0 * np.pi, size=num_images)
    return np.rint(
        np.column_stack((np.sin(angles), np.cos(angles))) * distances[:, None]
    ).astype(np.int32)


def shift_objects(y_true, class_ids, mean):
    """Shift selected connected objects using one sampled vector per image."""
    y_true = np.asarray(y_true)
    if y_true.ndim != 3:
        raise ValueError("'y_true' must have shape (images, height, width).")
    if isinstance(class_ids, (int, np.integer)):
        class_ids = [int(class_ids)]
    class_ids = [int(class_id) for class_id in class_ids]

    shift_vectors = _sample_shift_vectors(len(y_true), mean)
    output = y_true.copy()
    structure = np.ones((3, 3), dtype=np.int32)
    neighbor_structure = np.ones((3, 3), dtype=bool)

    for image_index, (original, (dy, dx)) in enumerate(
        zip(y_true, shift_vectors)
    ):
        if dx == 0 and dy == 0:
            continue

        shifted_image = output[image_index].copy()
        height, width = original.shape
        shifted_occupancy = np.zeros_like(original, dtype=bool)

        for class_id in class_ids:
            labeled, _ = ndimage.label(original == class_id, structure=structure)

            for object_id, object_slice in enumerate(
                ndimage.find_objects(labeled), start=1
            ):
                if object_slice is None:
                    continue

                y_slice, x_slice = object_slice
                y0, y1 = y_slice.start, y_slice.stop
                x0, x1 = x_slice.start, x_slice.stop
                object_patch = labeled[object_slice] == object_id
                object_rows, object_cols = np.nonzero(object_patch)
                original_rows = object_rows + y0
                original_cols = object_cols + x0
                origin_labels = shifted_image[original_rows, original_cols].copy()
                occupied_by_prior_shift = shifted_occupancy[
                    original_rows, original_cols
                ].copy()
                new_rows = original_rows + int(dy)
                new_cols = original_cols + int(dx)
                in_bounds = (
                    (new_rows >= 0)
                    & (new_rows < height)
                    & (new_cols >= 0)
                    & (new_cols < width)
                )

                shifted_image[original_rows, original_cols] = 0
                shifted_image[new_rows[in_bounds], new_cols[in_bounds]] = class_id
                shifted_occupancy[new_rows[in_bounds], new_cols[in_bounds]] = True

                new_on_original = np.zeros_like(object_patch, dtype=bool)
                inside_original_box = (
                    in_bounds
                    & (new_rows >= y0)
                    & (new_rows < y1)
                    & (new_cols >= x0)
                    & (new_cols < x1)
                )
                new_on_original[
                    new_rows[inside_original_box] - y0,
                    new_cols[inside_original_box] - x0,
                ] = True
                vacated_patch = object_patch & ~new_on_original
                if not vacated_patch.any():
                    continue

                vacated_pixels = vacated_patch[object_rows, object_cols]
                restore_prior_shift = vacated_pixels & occupied_by_prior_shift
                shifted_image[
                    original_rows[restore_prior_shift],
                    original_cols[restore_prior_shift],
                ] = origin_labels[restore_prior_shift]

                needs_neighbor_fill = vacated_pixels & ~occupied_by_prior_shift
                if not needs_neighbor_fill.any():
                    continue

                padded_slice = (
                    slice(max(y0 - 1, 0), min(y1 + 1, height)),
                    slice(max(x0 - 1, 0), min(x1 + 1, width)),
                )
                local_object = labeled[padded_slice] == object_id
                neighbor_mask = (
                    ndimage.binary_dilation(
                        local_object, structure=neighbor_structure
                    )
                    & ~local_object
                )
                neighbor_labels = shifted_image[padded_slice][neighbor_mask]
                valid_labels = neighbor_labels[neighbor_labels != class_id]
                if valid_labels.size:
                    fill_class = int(
                        np.bincount(valid_labels.astype(np.int64)).argmax()
                    )
                    shifted_image[
                        original_rows[needs_neighbor_fill],
                        original_cols[needs_neighbor_fill],
                    ] = fill_class

        output[image_index] = shifted_image

    return output


def noisy_zoom(y_true, mean):
    """Zoom masks in or out by a normally distributed magnitude."""
    y_true = np.asarray(y_true)
    if y_true.ndim != 3:
        raise ValueError("'y_true' must have shape (images, height, width).")
    if not np.isfinite(mean) or not 0.0 < mean < 1.0:
        raise ValueError("'mean' must be finite and strictly between zero and one.")

    output = np.empty_like(y_true)
    h, w = y_true.shape[1], y_true.shape[2]
    center = (w / 2.0, h / 2.0)
    std = 0.1 * mean
    magnitudes = np.random.normal(loc=mean, scale=std, size=len(y_true))
    invalid = (magnitudes <= 0.0) | (magnitudes >= 1.0)
    while invalid.any():
        magnitudes[invalid] = np.random.normal(
            loc=mean, scale=std, size=int(invalid.sum())
        )
        invalid = (magnitudes <= 0.0) | (magnitudes >= 1.0)
    signs = 2 * np.random.binomial(1, 0.5, size=len(y_true)) - 1
    scales = 1.0 + signs * magnitudes

    for i, (mask, scale) in enumerate(zip(y_true, scales)):
        matrix = cv2.getRotationMatrix2D(center, 0.0, scale)

        if scale >= 1.0:
            border_mode = cv2.BORDER_CONSTANT
        else:   
            border_mode = cv2.BORDER_REFLECT_101

        output[i] = cv2.warpAffine(
            mask,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=border_mode,
        )

    return output


def shift_scene(y_true, mean):
    """Shift each full label mask in an independently sampled direction."""
    y_true = np.asarray(y_true)
    if y_true.ndim != 3:
        raise ValueError("'y_true' must have shape (images, height, width).")

    shift_vectors = _sample_shift_vectors(len(y_true), mean)
    output = np.empty_like(y_true)

    for i, mask in enumerate(y_true):
        shift_y, shift_x = shift_vectors[i]
        output[i] = ndimage.shift(
            mask,
            shift=(shift_y, shift_x),
            order=0,
            mode="reflect",
            cval=0.0,
        )

    return output


def omission_noise(
    y_true,
    class_ids=None,
    omission_prob=0.25,
    small_object_threshold=None,
    fallback_class=0,
):
    """
    Simulate human annotation omission errors for selected small objects.
    """
    if small_object_threshold is None:
        raise ValueError("omission_noise requires a 'small_object_threshold'.")
    if not 0.0 <= omission_prob <= 1.0:
        raise ValueError("'omission_prob' must be between 0 and 1.")

    if class_ids is None:
        class_ids = np.arange(9)[1:]
    if isinstance(class_ids, (int, np.integer)):
        class_ids = [int(class_ids)]
    class_ids = [int(class_id) for class_id in class_ids]

    noisy = np.asarray(y_true).copy()
    structure = np.ones((3, 3), dtype=np.int32)

    for i, mask in enumerate(y_true):
        noisy_mask = noisy[i]
        for class_id in class_ids:
            class_mask = mask == class_id
            if not class_mask.any():
                continue

            labeled, n_objects = ndimage.label(class_mask, structure=structure)
            for obj_id in range(1, n_objects + 1):
                obj_mask = labeled == obj_id
                obj_size = obj_mask.sum()

                if obj_size > small_object_threshold:
                    continue

                if np.random.rand() < omission_prob:
                    neighbor_region = (
                        ndimage.binary_dilation(obj_mask, structure=structure) & ~obj_mask
                    )
                    neighbor_labels = noisy_mask[neighbor_region]

                    if neighbor_labels.size > 0:
                        valid_labels = neighbor_labels[neighbor_labels != class_id]
                        if valid_labels.size == 0:
                            valid_labels = neighbor_labels
                        fill_label = int(
                            np.bincount(valid_labels.astype(np.int64)).argmax()
                        )
                    else:
                        fill_label = fallback_class

                    noisy_mask[obj_mask] = fill_label

    return noisy


def small_object_threshold_percentile(y_true, class_ids=None, percentile=90):
    """
    Return a connected-component size percentile for selected classes.
    """
    if class_ids is None:
        class_ids = np.arange(9)[1:]
    if isinstance(class_ids, (int, np.integer)):
        class_ids = [int(class_ids)]
    class_ids = [int(class_id) for class_id in class_ids]

    structure = np.ones((3, 3), dtype=np.int32)
    object_sizes = []

    for mask in np.asarray(y_true):
        for class_id in class_ids:
            labeled, n_objects = ndimage.label(mask == class_id, structure=structure)
            if n_objects == 0:
                continue
            object_sizes.extend(np.bincount(labeled.ravel())[1:])

    if not object_sizes:
        raise ValueError(
            "Cannot compute small object threshold: no connected objects found "
            f"for class_ids={class_ids}."
        )

    return float(np.percentile(np.asarray(object_sizes, dtype=np.int64), percentile))


def noise_methods(n_classes):
    return {
        "swap_classes": add_semantic_swapping,
        "zoom": noisy_zoom,
        "shift_scene": shift_scene,
        "shift_objects": shift_objects,
        "add_omission": omission_noise,
        "random_pixels": add_random_pixel_noise,
        "boundary_morphology": boundary_dilation_erosion,
    }


def _load_sorted_masks(paths):
    return np.array([load_labels(path) for path in sorted(paths, key=lambda p: p.name)])


def _save_generated_masks(output, output_dir, source_files):
    for img, source_path in zip(output, source_files):
        Image.fromarray(img.astype(np.uint8)).save(output_dir / f"{source_path.stem}.png")


def add_noise(
    pr_path,
    n_classes,
    noise_specs,
    *,
    data_root=None,
    db_path=None,
):
    data_root = pathlib.Path(data_root) if data_root is not None else DATA_ROOT
    pr_path = pathlib.Path(pr_path)
    if not pr_path.is_absolute():
        pr_path = data_root / pr_path

    methods = noise_methods(n_classes)
    unknown_methods = set(noise_specs) - set(methods)
    if unknown_methods:
        known_methods = ", ".join(sorted(methods))
        unknown_text = ", ".join(sorted(unknown_methods))
        raise ValueError(f"Unknown noise method: {unknown_text}. Expected one of: {known_methods}")

    gt_files = sorted((data_root / "test" / "labels").glob("*.tif"), key=lambda p: p.name)
    y_true = _load_sorted_masks(gt_files)
    created_dirs = []

    for noise_method, noise_spec in noise_specs.items():
        generator = methods[noise_method]
        count = noise_spec.get("count", 0)
        base_params = dict(noise_spec.get("params", {}))

        for _ in range(count):
            gen_params = dict(base_params)
            if noise_method == "add_omission" and "small_object_threshold" not in gen_params:
                gen_params["small_object_threshold"] = small_object_threshold_percentile(
                    y_true,
                    class_ids=gen_params.get("class_ids"),
                    percentile=90,
                )

            timestamp_text = creation_tstamp()
            dataset_id = generated_directory_name(noise_method, timestamp=timestamp_text)
            output_dir = data_root / "noise" / noise_method / dataset_id

            if noise_method == "swap_classes":
                pr_files = sorted(pr_path.glob("*.png"), key=lambda p: p.name)
                if len(pr_files) != len(gt_files):
                    raise ValueError(
                        f"swap_classes requires {len(gt_files)} predictions, found {len(pr_files)} in {pr_path}"
                    )
                y_pred = _load_sorted_masks(pr_files)
                output = generator(y_true, y_pred, **gen_params)
            else:
                output = generator(y_true, **gen_params)

            output_dir.mkdir(parents=True)
            _save_generated_masks(output, output_dir, gt_files)
            # color_model_output(result_root=output_dir)
            log_generation(
                timestamp_text,
                generator.__name__,
                gen_params,
                db_path=db_path,
            )
            created_dirs.append(output_dir)

    return created_dirs
