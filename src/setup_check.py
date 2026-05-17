import importlib
import sys
import pkgutil
from pathlib import Path
import Semantic_Segmentation

def check_workspace_details(package_name):
    print(f"\n🔍 Checking Workspace Content: {package_name}")
    print("-" * 60)
    try:
        module = importlib.import_module(package_name)
        module_path = Path(module.__file__).parent
        
        print(f"📍 Physical Location: {module_path}")
        
        # Check if the path is actually inside your 'external' folder
        if "external" in str(module_path):
            print("✅ SUCCESS: Linked to the local 'external' folder.")
        else:
            print("⚠️  WARNING: Found in site-packages (Not using workspace!)")

        # List all .py files found inside that workspace folder
        print(f"\n📂 Python files found in {package_name}:")
        sub_modules = [name for _, name, _ in pkgutil.iter_modules([str(module_path)])]
        
        if not sub_modules:
            print("   (No sub-modules found. Is there an __init__.py?)")
        for sub in sub_modules:
            print(f"   ├── {sub}.py")
            
        return True
    except ImportError:
        print(f"❌ ERROR: Could not import '{package_name}'.")
        return False

def check_dependencies(deps):
    print(f"\n📦 Checking External Dependencies")
    print("-" * 60)
    for pkg, imp in deps:
        try:
            importlib.import_module(imp)
            print(f"✅ {pkg:<30} | Imported as: {imp}")
        except ImportError:
            print(f"❌ {pkg:<30} | NOT FOUND")

if __name__ == "__main__":
    # 1. Verify the Workspace link and its internal files
    # Ensure 'oem_sar' matches the folder/package name in your workspace
    workspace_ok = check_workspace_details(r"external/oem_sar")

    # 2. Verify the heavy hitters from your list
    dependencies = [
        ("albumentations", "albumentations"),
        ("opencv-python", "cv2"),
        ("rasterio", "rasterio"),
        ("segmentation-models-pytorch", "segmentation_models_pytorch"),
        ("torch", "torch"),
    ]
    check_dependencies(dependencies)

    if workspace_ok:
        print("\n🚀 Workspace is live. Changes to files in 'external/' will reflect immediately.")
