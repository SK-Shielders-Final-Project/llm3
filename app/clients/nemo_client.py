import os
import logging
import asyncio
from typing import Any
from nemoguardrails import RailsConfig, LLMRails
from app.clients.aws_guardrail_client import GuardrailDecision

class NemoClient:
    """
    Unified NVIDIA NeMo Guardrails client.
    Uses the official nemoguardrails library with local .co files and config.yml.
    Backend LLM engine is configured in config.yml (e.g., NVIDIA NIM).
    """

    def __init__(self, config_path: str):
        self._logger = logging.getLogger("guardrail_client")
        print(f"[DEBUG] NemoClient.__init__ with config_path: {config_path}")
        try:
            # Move imports inside to catch missing dependency errors during build
            print("[DEBUG] Importing nemoguardrails...")
            from nemoguardrails import RailsConfig, LLMRails
            
            # Load configuration from the directory containing config.yml and .co files
            print(f"[DEBUG] Loading RailsConfig from {config_path}...")
            self.config = RailsConfig.from_path(config_path)
            
            # Initialize rails - this will load the configured LLMS
            print("[DEBUG] Initializing LLMRails...")
            self.rails = LLMRails(self.config)
            
            self._logger.info(f"NemoClient initialized with config from: {config_path}")
            print("[DEBUG] NemoClient initialization complete")
        except ImportError as e:
            print(f"[DEBUG] nemoguardrails library NOT FOUND: {e}")
            self._logger.error(f"nemoguardrails library NOT FOUND: {e}")
            raise
        except Exception as e:
            print(f"[DEBUG] Failed to initialize NemoClient: {e}")
            self._logger.error(f"Failed to initialize NemoClient: {e}")
            raise

    def apply(self, *, text: str, source: str) -> GuardrailDecision:
        """
        Applies NEMO guardrails to the given text.
        source is either "INPUT" or "OUTPUT".
        """
        if not text:
            return GuardrailDecision(action="NONE", output_text=text, raw={})

        print(f"[DEBUG] NemoClient.apply source={source} text_len={len(text)}")
        try:
            # NeMo Guardrails library is primarily async. 
            # We wrap it in a sync call for compatibility with the current orchestrator.
            
            import asyncio
            try:
                # Try to use existing loop if available (FastAPI might have one)
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            print("[DEBUG] Running rails generate_async...")
            if loop.is_running():
                # If we are in an async environment, we might need a different approach
                # but for simplicity in a sync handler:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.rails.generate_async(prompt=text))
                    response_text = future.result()
            else:
                response_text = loop.run_until_complete(self.rails.generate_async(prompt=text))

            print(f"[DEBUG] Rails response: {response_text[:100]}...")
            
            action = "NONE"
            if response_text != text:
                refusal_keywords = ["죄송합니다", "보안 정책", "처리할 수 없", "도움 주기 어렵", "안전한 범위"]
                if any(kw in response_text for kw in refusal_keywords) or not response_text.strip():
                    action = "BLOCK"
                    self._logger.warning(f"NeMo Guardrail BLOCKED content source={source}. Verdict: {response_text}")
                    print(f"[DEBUG] Action: BLOCK (refusal detected)")
                else:
                    print(f"[DEBUG] Action: NONE (text modified but not refusal)")

            return GuardrailDecision(
                action=action,
                output_text=response_text,
                raw={"provider": "nemo_unified", "source": source, "original_text": text}
            )
        except Exception as e:
            print(f"[DEBUG] NemoClient.apply error: {e}")
            self._logger.exception(f"NemoClient apply failed: {e}")
            return GuardrailDecision(action="NONE", output_text=text, raw={"error": str(e)})

def build_nemo_client() -> NemoClient | None:
    """Factory to build the unified NemoClient."""
    print("[DEBUG] build_nemo_client() called")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    nemo_config_dir = os.path.join(base_dir, "clients", "NEMO")
    
    # Ensure NVIDIA_API_KEY is present if using NIM engine
    if not os.getenv("NVIDIA_API_KEY"):
        print("[DEBUG] WARNING: NVIDIA_API_KEY is missing")
        logging.getLogger("guardrail_client").warning("NVIDIA_API_KEY is missing. NemoClient might fail to initialize.")

    if os.path.exists(nemo_config_dir):
        try:
            print(f"[DEBUG] Config directory found: {nemo_config_dir}")
            return NemoClient(nemo_config_dir)
        except Exception as e:
            print(f"[DEBUG] Exception during NemoClient construction: {e}")
            logging.getLogger("guardrail_client").error(f"Failed to build NemoClient: {e}")
            return None
    
    print(f"[DEBUG] Config directory NOT FOUND: {nemo_config_dir}")
    return None

