import os
import logging
from typing import Any
from nemoguardrails import RailsConfig, LLMRails
from app.clients.aws_guardrail_client import GuardrailDecision

class NemoLibraryClient:
    """Official NeMo Guardrails library client that uses local Colang files."""
    
    def __init__(self, config_path: str):
        self._logger = logging.getLogger("guardrail_client")
        try:
            # NEMO config directory (contains config.yml and .co files)
            self.config = RailsConfig.from_path(config_path)
            self.rails = LLMRails(self.config)
            self._logger.info(f"NeMo Library Client initialized with config from: {config_path}")
        except Exception as e:
            self._logger.error(f"Failed to initialize NeMo Library Client: {e}")
            raise

    def apply(self, *, text: str, source: str) -> GuardrailDecision:
        """Applies rails using the official library."""
        if not text:
            return GuardrailDecision(action="NONE", output_text=text, raw={})
            
        try:
            # Identify if it's INPUT or OUTPUT for appropriate rail execution
            # LLMRails.generate is the main entry point
            # For a simple 'apply' like interface, we can use the chat API internally
            
            # Note: This is a synchronous wrapper for what is typically an async library
            # In a real production app, we'd want to use async/await
            import asyncio
            
            async def _run_rails(input_text):
                return await self.rails.generate_async(prompt=input_text)
            
            response_text = asyncio.run(_run_rails(text))
            
            # NeMo Guardrails library typically returns the 'filtered' text
            # If the text was blocked/altered by a flow, it will differ from original
            action = "NONE"
            if response_text != text:
                # If there's a specific "stop" or refusal, we might consider it a BLOCK
                # NeMo library often replaces with predefined bot messages
                action = "BLOCK" if "죄송합니다" in response_text or not response_text else "NONE"
                
            return GuardrailDecision(
                action=action,
                output_text=response_text,
                raw={"provider": "nemo_library", "source": source}
            )
        except Exception as e:
            self._logger.exception(f"NeMo Library apply failed: {e}")
            return GuardrailDecision(action="NONE", output_text=text, raw={"error": str(e)})

def build_nemo_library_client() -> NemoLibraryClient | None:
    """Factory to build the library-based client."""
    # Path to app/clients/NEMO directory
    base_dir = os.path.dirname(os.path.dirname(__file__))
    nemo_config_dir = os.path.join(base_dir, "clients", "NEMO")
    
    if os.path.exists(nemo_config_dir):
        try:
            return NemoLibraryClient(nemo_config_dir)
        except Exception:
            return None
    return None
