# Fullstack PDF App with FastAPI and Next.js

For quick output of the demo:

<p align="center">
    <img src="https://github.com/hirenhk15/langchain-plus-pdf-app/tree/master/frontend/pdf-app/public/app_demo.png" />
</p>

### 1. Run the backend:
- Navigate to /backend and run following command in the terminal:

    ```uvicorn main:app --reload```

- For production, add following commant to start the backend app:
    ```uvicorn main:app --host 0.0.0.0 --port $PORT --workers 4```

### 2. Run the frontend:
 - Navigate to /frontend/todo-app and run following command in the terminal:

    ```npm run dev```
