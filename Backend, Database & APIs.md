# Member 3 — Backend, Database & APIs

## Role
Backend service, database, API layer, validation, error handling, and integration between frontend and ML/data modules.

## Recommended Stack
- Python
- FastAPI
- PostgreSQL/Supabase
- Pydantic
- Environment variables for secrets

## Tasks

### 1. Database
Design entities/tables for:
- Wells
- Groundwater observations
- Environmental data
- Predictions
- Recharge results
- Risk assessments
- XAI explanations
- Alerts

Use clear primary/foreign keys.

### 2. Groundwater APIs
```text
GET /api/wells
GET /api/wells/{well_id}
GET /api/wells/{well_id}/observations
```

### 3. Prediction APIs
```text
GET /api/predictions
GET /api/predictions/{well_id}
```

### 4. Recharge APIs
```text
GET /api/recharge
GET /api/recharge/{well_id}
```

### 5. Risk APIs
```text
GET /api/risk
GET /api/risk/{well_id}
```

### 6. XAI API
```text
GET /api/explanations/{well_id}
```

Return feature, contribution, direction, importance, and model/version information as appropriate.

### 7. ML Integration
Provide a clean interface for Member 4's prediction/recharge/XAI outputs. Do not duplicate ML logic in the frontend.

### 8. Security
- Keep secrets in `.env`.
- Never commit API/database credentials.
- Configure CORS.
- Validate inputs.
- Add logging and error handling.

### 9. API Documentation
Maintain stable response schemas agreed with Members 1, 2, and 4.

### 10. Testing
Test valid requests, invalid IDs, empty results, malformed parameters, database failures, and ML/API failures.

## Deliverables
FastAPI backend, database schema, groundwater/prediction/recharge/risk/XAI APIs, validation, documentation, tests, and environment configuration.
