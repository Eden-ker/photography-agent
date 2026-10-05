import threading
import queue
import time
from typing import Optional, Tuple
from core.scene import SceneState
from core.task import CompositionTarget
import numpy as np

class AgentWorker:
    """
    A background worker that runs the VLM agent on a separate thread.
    """
    def __init__(self, agent):
        self.agent = agent
        self.request_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self._is_busy = False
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        
    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                frame, state = self.request_queue.get(timeout=0.1)
                self._is_busy = True
                try:
                    # Analyze frame using the VLM agent with a bounded retry for transient errors (e.g. 503)
                    target = None
                    for attempt in range(3):
                        target = self.agent.analyze(frame, state)
                        if target is not None:
                            break
                        print(f"[AgentWorker] Analysis failed (attempt {attempt + 1}/3). Retrying in 1s...")
                        time.sleep(1.0)
                        
                    while not self.result_queue.empty():
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            break
                            
                    self.result_queue.put(target)
                finally:
                    self._is_busy = False
                    self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self._is_busy = False
                print(f"[AgentWorker] Unhandled error: {e}")
                # Put a None result so the controller knows it failed
                try:
                    self.result_queue.put_nowait(None)
                except:
                    pass
                
    def request_analysis(self, frame: np.ndarray, state: SceneState) -> bool:
        if self._is_busy or self.request_queue.full():
            return False
            
        try:
            self.request_queue.put_nowait((frame.copy(), state))
            return True
        except queue.Full:
            return False
                
    def get_result(self) -> Tuple[bool, Optional[CompositionTarget]]:
        try:
            task = self.result_queue.get_nowait()
            return True, task
        except queue.Empty:
            return False, None
            
    def shutdown(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
