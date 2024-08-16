import config
from functools import lru_cache
from typing import Union

from routers import pdfs
from fastapi import FastAPI, Depends
from fastapi.responses import PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# CORS configuration, needed for frontend development

# origins = [
#     "http://localhost:3000",
#     "https://pdf-app-fastapi-vercel-fullstack.vercel.app/",
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Add routers
app.include_router(pdfs.router)

# Global http exception handler, to handle errors
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    print(f"{repr(exc)}")
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

# To use the settings
@lru_cache()
def get_settings():
    return config.Settings()

@app.get("/")
def read_root(settings: config.Settings = Depends(get_settings)):
    # Print the app_name configuration
    print(settings.app_name)
    return "Hello PDF World!"