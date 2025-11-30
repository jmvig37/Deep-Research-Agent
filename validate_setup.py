"""Simple validation script to check if the setup is correct."""
import sys
import importlib.util

def check_imports():
    """Check if all required packages can be imported."""
    required_packages = [
        'langgraph',
        'langchain',
        'langchain_openai',
        'langchain_community',
        'pydantic',
        'pydantic_settings',
        'dotenv'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'dotenv':
                importlib.import_module('dotenv')
            else:
                importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    return missing

def check_files():
    """Check if required files exist."""
    import os
    
    required_files = [
        'agent.py',
        'config.py',
        'main.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            missing.append(file)
    
    return missing

def check_env_file():
    """Check if .env file exists."""
    import os
    
    if os.path.exists('.env'):
        print("✓ .env file exists")
        return True
    else:
        print("⚠ .env file not found (create it from .env.example)")
        return False

if __name__ == "__main__":
    print("Validating Deep Research Agent Setup")
    print("=" * 50)
    
    print("\nChecking required packages:")
    missing_packages = check_imports()
    
    print("\nChecking required files:")
    missing_files = check_files()
    
    print("\nChecking environment:")
    env_exists = check_env_file()
    
    print("\n" + "=" * 50)
    if missing_packages or missing_files:
        print("\n❌ Setup incomplete!")
        if missing_packages:
            print(f"\nMissing packages: {', '.join(missing_packages)}")
            print("Run: pip install -r requirements.txt")
        if missing_files:
            print(f"\nMissing files: {', '.join(missing_files)}")
        sys.exit(1)
    else:
        print("\n✓ Setup looks good!")
        if not env_exists:
            print("\n⚠ Remember to create a .env file with your API keys")
        print("\nYou can now run: python main.py 'your research query'")

