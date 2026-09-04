import asyncio
import os
import httpx
from typing import Optional

class KeepAliveService:
    """
    Self-ping background service running every 10 minutes (600,000 ms)
    to keep cloud containers (Render Free Tier) active and prevent idle spin-down.
    Equivalent to the velocura KeepAliveService heartbeat.
    """
    def __init__(self, port: int = 8000, interval_seconds: int = 600, initial_delay_seconds: int = 60):
        self.port = port
        self.interval_seconds = interval_seconds
        self.initial_delay_seconds = initial_delay_seconds
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

    def start(self):
        if not self._running:
            self._running = True
            try:
                loop = asyncio.get_running_loop()
                self._task = loop.create_task(self._run_loop())
            except RuntimeError:
                # In case event loop is not yet running
                pass

    def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _run_loop(self):
        # Initial delay to give the server time to fully start
        await asyncio.sleep(self.initial_delay_seconds)
        while self._running:
            await self.ping_self()
            await asyncio.sleep(self.interval_seconds)

    async def ping_self(self):
        """
        Pings the public Render URL if defined (triggering incoming HTTP traffic)
        or falls back to localhost health check.
        """
        external_url = (
            os.getenv("RENDER_EXTERNAL_URL")
            or os.getenv("PING_URL")
            or os.getenv("KEEP_ALIVE_URL")
        )
        
        target_urls = []
        if external_url:
            clean_url = external_url.rstrip("/")
            target_urls.append(f"{clean_url}/api/health")
        
        # Localhost fallback
        target_urls.append(f"http://127.0.0.1:{self.port}/api/health")

        for url in target_urls:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(url)
                    print(f"🔄 RENDER KEEP-ALIVE HEARTBEAT: Self-ping {url} [HTTP {response.status_code}]")
                    if response.status_code == 200:
                        break
            except Exception:
                print(f"🔄 RENDER KEEP-ALIVE HEARTBEAT: Heartbeat active ({url}).")
