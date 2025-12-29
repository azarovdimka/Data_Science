"""
WebSocket для передачи прогресса обработки
"""

from fastapi import WebSocket
import json
import asyncio

class ProgressManager:
    def __init__(self):
        self.connections = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)
    
    async def send_progress(self, percent: int, message: str):
        """Отправка прогресса всем подключенным клиентам"""
        if self.connections:
            data = {
                "type": "progress",
                "percent": percent,
                "message": message
            }
            
            # Отправляем всем подключенным клиентам
            disconnected = []
            for connection in self.connections:
                try:
                    await connection.send_text(json.dumps(data))
                except:
                    disconnected.append(connection)
            
            # Удаляем отключенные соединения
            for conn in disconnected:
                self.disconnect(conn)

# Глобальный менеджер прогресса
progress_manager = ProgressManager()