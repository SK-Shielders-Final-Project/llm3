
import os
import sys
import asyncio
import logging

# Add current directory to sys.path
sys.path.append(os.getcwd())

from nemoguardrails import RailsConfig, LLMRails

def test_load_nemo():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test_nemo")
    
    nemo_config_dir = os.path.join(os.getcwd(), "app", "clients", "NEMO")
    print(f"Checking directory: {nemo_config_dir}")
    if not os.path.exists(nemo_config_dir):
        print(f"Error: Directory {nemo_config_dir} does not exist")
        return

    print("Files in NEMO directory:")
    for f in os.listdir(nemo_config_dir):
        print(f" - {f}")

    try:
        print("\nAttempting to load RailsConfig...")
        config = RailsConfig.from_path(nemo_config_dir)
        print("RailsConfig loaded successfully")
        
        print("\nAttempting to initialize LLMRails...")
        rails = LLMRails(config)
        print("LLMRails initialized successfully")
        
        async def run_test():
            test_text = "import os; print(os.name)"
            print(f"\nTesting prompt: {test_text}")
            # Use generate_async which is the standard way
            result = await rails.generate_async(prompt=test_text)
            print(f"Result: {result}")
            
        asyncio.run(run_test())
        
    except Exception as e:
        print(f"\nError during Nemo initialization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load_nemo()
