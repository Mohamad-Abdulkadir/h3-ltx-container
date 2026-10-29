from typing import Any


class RuntimeFault(Exception):
    def __init__(self, code: str, message: str, status: int = 422, **details: Any):
        super().__init__(message)
        self.code, self.message, self.status, self.details = code, message, status, details

    def payload(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}
