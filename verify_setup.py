"""
Verify EpiWatch Installation and Setup
Run this to check if everything is properly installed
"""

import sys
import importlib
from colorama import init, Fore, Style

init(autoreset=True)

def check_import(module_name, display_name=None):
    """Check if a module can be imported"""
    if display_name is None:
        display_name = module_name
    
    try:
        importlib.import_module(module_name)
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} {display_name}")
        return True
    except ImportError:
        print(f"{Fore.RED}✗{Style.RESET_ALL} {display_name} - NOT INSTALLED")
        return False

def main():
    print("\n" + "="*60)
    print("  EpiWatch Installation Verification")
    print("="*60 + "\n")
    
    # Core dependencies
    print("Core Dependencies:")
    print("-" * 60)
    core_deps = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('torch', 'PyTorch'),
        ('transformers', 'Hugging Face Transformers'),
        ('sklearn', 'scikit-learn'),
    ]
    
    core_ok = all(check_import(module, name) for module, name in core_deps)
    
    # NLP dependencies
    print("\nNLP Libraries:")
    print("-" * 60)
    nlp_deps = [
        ('nltk', 'NLTK'),
        ('langdetect', 'langdetect'),
    ]
    
    nlp_ok = all(check_import(module, name) for module, name in nlp_deps)
    
    # API dependencies
    print("\nAPI & Backend:")
    print("-" * 60)
    api_deps = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('pydantic', 'Pydantic'),
    ]
    
    api_ok = all(check_import(module, name) for module, name in api_deps)
    
    # Visualization
    print("\nVisualization:")
    print("-" * 60)
    viz_deps = [
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
    ]
    
    viz_ok = all(check_import(module, name) for module, name in viz_deps)
    
    # Statistics
    print("\nStatistics & ML:")
    print("-" * 60)
    stat_deps = [
        ('scipy', 'SciPy'),
        ('statsmodels', 'Statsmodels'),
    ]
    
    stat_ok = all(check_import(module, name) for module, name in stat_deps)
    
    # Check Python version
    print("\nPython Environment:")
    print("-" * 60)
    python_version = sys.version.split()[0]
    print(f"Python Version: {python_version}")
    
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 8:
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} Python version is compatible (3.8+)")
        python_ok = True
    else:
        print(f"{Fore.RED}✗{Style.RESET_ALL} Python 3.8+ required, found {major}.{minor}")
        python_ok = False
    
    # Check CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} CUDA is available (GPU acceleration enabled)")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"{Fore.YELLOW}⚠{Style.RESET_ALL} CUDA not available (will use CPU)")
    except:
        print(f"{Fore.YELLOW}⚠{Style.RESET_ALL} Could not check CUDA availability")
    
    # Check NLTK data
    print("\nNLTK Data:")
    print("-" * 60)
    try:
        import nltk
        required_data = ['punkt', 'stopwords', 'wordnet']
        nltk_ok = True
        
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}' if data_name == 'punkt' else f'corpora/{data_name}')
                print(f"{Fore.GREEN}✓{Style.RESET_ALL} {data_name}")
            except LookupError:
                print(f"{Fore.RED}✗{Style.RESET_ALL} {data_name} - NOT DOWNLOADED")
                nltk_ok = False
    except:
        print(f"{Fore.RED}✗{Style.RESET_ALL} Could not check NLTK data")
        nltk_ok = False
    
    # Check project structure
    print("\nProject Structure:")
    print("-" * 60)
    
    import os
    required_dirs = [
        'data/raw',
        'data/processed',
        'models/saved',
        'src/models',
        'src/evaluation',
        'src/preprocessing',
        'src/api',
        'notebooks',
        'outputs/alerts',
        'outputs/visualizations'
    ]
    
    structure_ok = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} {dir_path}")
        else:
            print(f"{Fore.RED}✗{Style.RESET_ALL} {dir_path} - MISSING")
            structure_ok = False
    
    # Final summary
    print("\n" + "="*60)
    print("  Summary")
    print("="*60 + "\n")
    
    all_ok = all([core_ok, nlp_ok, api_ok, viz_ok, stat_ok, python_ok, structure_ok])
    
    if all_ok:
        print(f"{Fore.GREEN}✓ All checks passed!{Style.RESET_ALL}")
        print(f"\n{Fore.GREEN}You're ready to start training models!{Style.RESET_ALL}")
        print("\nNext steps:")
        print("  1. Run: python src/models/train_all.py")
        print("  2. Or open: notebooks/epiwatch_training.ipynb")
    else:
        print(f"{Fore.YELLOW}⚠ Some components are missing{Style.RESET_ALL}")
        print("\nTo fix:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Download NLTK data: python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')\"")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{Fore.RED}Error during verification: {str(e)}{Style.RESET_ALL}\n")
        import traceback
        traceback.print_exc()
