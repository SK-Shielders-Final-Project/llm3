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
        self._logger.info(f"[NEMO-DEBUG] NemoClient.__init__ with config_path: {config_path}")
        try:
            # Move imports inside to catch missing dependency errors during build
            self._logger.info("[NEMO-DEBUG] Importing nemoguardrails...")
            from nemoguardrails import RailsConfig, LLMRails
            
            # Ensure OPENAI_API_KEY is available if we're using the 'openai' engine for NIM
            # NVIDIA NIM API is OpenAI-compatible and often we use the 'openai' engine
            # but with a different base_url.
            if not os.getenv("OPENAI_API_KEY") and os.getenv("NVIDIA_API_KEY"):
                self._logger.info("[NEMO-DEBUG] Setting OPENAI_API_KEY from NVIDIA_API_KEY for compatibility")
                os.environ["OPENAI_API_KEY"] = os.environ["NVIDIA_API_KEY"]

            # Load configuration from the directory containing config.yml and .co files
            abs_config_path = os.path.abspath(config_path)
            self._logger.info(f"[NEMO-DEBUG] Loading RailsConfig from {abs_config_path}...")
            self.config = RailsConfig.from_path(abs_config_path)
            
            # Initialize rails - this will load the configured LLMS
            self._logger.info("[NEMO-DEBUG] Initializing LLMRails...")
            self.rails = LLMRails(self.config)
            
            self._logger.info(f"NemoClient initialized with config from: {abs_config_path}")
            self._logger.info("[NEMO-DEBUG] NemoClient initialization complete")
        except ImportError as e:
            self._logger.error(f"[NEMO-DEBUG] nemoguardrails library NOT FOUND: {e}")
            raise
        except Exception as e:
            self._logger.error(f"[NEMO-DEBUG] Failed to initialize NemoClient: {e}")
            raise

    def apply(self, *, text: str, source: str) -> GuardrailDecision:
        """
        Applies NEMO guardrails to the given text.
        source is either "INPUT" or "OUTPUT".
        """
        if not text:
            return GuardrailDecision(action="NONE", output_text=text, raw={})

        self._logger.info(f"[NEMO-DEBUG] NemoClient.apply source={source} text_len={len(text)}")
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
            
            self._logger.info("[NEMO-DEBUG] Running rails generate_async...")
            
            # If we are in an async environment, we might need a different approach
            # Using ThreadPoolExecutor is a common way to run async code in a sync block
            if loop.is_running():
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.rails.generate_async(prompt=text))
                    response_text = future.result()
            else:
                response_text = loop.run_until_complete(self.rails.generate_async(prompt=text))

            self._logger.info(f"[NEMO-DEBUG] Rails response received (first 50 chars): {response_text[:50]}...")
            
            action = "NONE"
            if response_text != text:
                refusal_keywords = ["죄송합니다", "보안 정책", "처리할 수 없", "도움 주기 어렵", "안전한 범위"]
                if any(kw in response_text for kw in refusal_keywords) or not response_text.strip():
                    action = "BLOCK"
                    self._logger.warning(f"NeMo Guardrail BLOCKED content source={source}. Verdict: {response_text}")
                    self._logger.info(f"[NEMO-DEBUG] Action: BLOCK (refusal detected)")
                else:
                    self._logger.info(f"[NEMO-DEBUG] Action: NONE (text modified but not refusal)")

            return GuardrailDecision(
                action=action,
                output_text=response_text,
                raw={"provider": "nemo_unified", "source": source, "original_text": text}
            )
        except Exception as e:
            self._logger.error(f"[NEMO-DEBUG] NemoClient.apply error: {e}")
            self._logger.exception(f"NemoClient apply failed: {e}")
            return GuardrailDecision(action="NONE", output_text=text, raw={"error": str(e)})

def build_nemo_client() -> NemoClient | None:
    """Factory to build the unified NemoClient."""
    logger = logging.getLogger("guardrail_client")
    logger.info("[NEMO-DEBUG] build_nemo_client() called")
    
    # Get the absolute path to the app/clients/NEMO directory
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    nemo_config_dir = os.path.join(current_file_dir, "NEMO")
    
    # Ensure NVIDIA_API_KEY is present if using NIM engine
    if not os.getenv("NVIDIA_API_KEY"):
        logger.warning("[NEMO-DEBUG] NVIDIA_API_KEY is missing. NemoClient might fail to initialize.")

    if os.path.exists(nemo_config_dir):
        try:
            logger.info(f"[NEMO-DEBUG] Config directory found: {nemo_config_dir}")
            return NemoClient(nemo_config_dir)
        except Exception as e:
            logger.error(f"[NEMO-DEBUG] Exception during NemoClient construction: {e}")
            return None
    
    logger.error(f"[NEMO-DEBUG] Config directory NOT FOUND: {nemo_config_dir}")
    return None

