"""WebSocket endpoint for streaming detection events."""

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket, request: Request):
    manager = getattr(request.app.state, "ws_manager", None)
    if manager is None:
        await websocket.close(code=1011)
        return

    await manager.connect(websocket)

    try:
        # Keep the connection alive; clients may send pings or no data at all
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket)
