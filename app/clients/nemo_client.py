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
        try:
            # Load configuration from the directory containing config.yml and .co files
            self.config = RailsConfig.from_path(config_path)
            # Initialize rails - this will load the configured LLMS
            self.rails = LLMRails(self.config)
            self._logger.info(f"NemoClient initialized with config from: {config_path}")
        except Exception as e:
            self._logger.error(f"Failed to initialize NemoClient: {e}")
            raise

    def apply(self, *, text: str, source: str) -> GuardrailDecision:
        """
        Applies NEMO guardrails to the given text.
        source is either "INPUT" or "OUTPUT".
        """
        if not text:
            return GuardrailDecision(action="NONE", output_text=text, raw={})

        try:
            # NeMo Guardrails library is primarily async. 
            # We wrap it in a sync call for compatibility with the current orchestrator.
            # In a production environment, it is better to use async throughout.
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response_text = loop.run_until_complete(self.rails.generate_async(prompt=text))
            finally:
                loop.close()

            # Determine if the content was blocked.
            # NeMo Guardrails typically returns a refusal message (e.g., "죄송합니다...") 
            # or an empty string if it's blocked.
            action = "NONE"
            
            # Simple check: if the response changed significantly or contains refusal keywords
            # For "INPUT" rails, NeMo often returns the refusal text as the 'generation'
            if response_text != text:
                # If it contains refusal keywords or is a known bot refusal message
                refusal_keywords = ["죄송합니다", "보안 정책", "처리할 수 없", "도움 주기 어렵"]
                if any(kw in response_text for kw in refusal_keywords) or not response_text.strip():
                    action = "BLOCK"
                    self._logger.warning(f"NeMo Guardrail BLOCKED content source={source}. Verdict: {response_text}")
                else:
                    # It might be a regular correction/formatting rail, which we might not want to BLOCK
                    # but for security we might still want to be cautious.
                    # Here we follow the logic that if it's not a refusal, it's just modified text.
                    action = "NONE"

            return GuardrailDecision(
                action=action,
                output_text=response_text,
                raw={"provider": "nemo_unified", "source": source, "original_text": text}
            )
        except Exception as e:
            self._logger.exception(f"NemoClient apply failed: {e}")
            # Fallback to allow if guardrail fails, to avoid breaking the service,
            # but this behavior might need to be tightened for high security.
            return GuardrailDecision(action="NONE", output_text=text, raw={"error": str(e)})

def build_nemo_client() -> NemoClient | None:
    """Factory to build the unified NemoClient."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    nemo_config_dir = os.path.join(base_dir, "clients", "NEMO")
    
    # Ensure NVIDIA_API_KEY is present if using NIM engine
    if not os.getenv("NVIDIA_API_KEY"):
        logging.getLogger("guardrail_client").warning("NVIDIA_API_KEY is missing. NemoClient might fail to initialize.")

    if os.path.exists(nemo_config_dir):
        try:
            return NemoClient(nemo_config_dir)
        except Exception as e:
            logging.getLogger("guardrail_client").error(f"Failed to build NemoClient: {e}")
            return None
    return None
