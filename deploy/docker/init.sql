-- World Model Lens Database Schema

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Analysis jobs table
CREATE TABLE analysis_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    model_checkpoint VARCHAR(512) NOT NULL,
    analysis_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    results JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    processing_time_seconds FLOAT,
    CONSTRAINT valid_status CHECK (status IN ('pending', 'running', 'completed', 'failed'))
);

CREATE INDEX idx_analysis_jobs_status ON analysis_jobs(status);
CREATE INDEX idx_analysis_jobs_created_at ON analysis_jobs(created_at DESC);
CREATE INDEX idx_analysis_jobs_user_id ON analysis_jobs(user_id);

-- Benchmark results table
CREATE TABLE benchmark_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_checkpoint VARCHAR(512) NOT NULL,
    dataset_path VARCHAR(512) NOT NULL,
    benchmark_type VARCHAR(50) NOT NULL,
    metrics JSONB NOT NULL,
    profiles JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_benchmark_results_created_at ON benchmark_results(created_at DESC);

-- Model registry table
CREATE TABLE model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) UNIQUE NOT NULL,
    backend VARCHAR(50) NOT NULL,
    checkpoint_path VARCHAR(512) NOT NULL,
    config JSONB NOT NULL,
    metadata JSONB,
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_model_registry_backend ON model_registry(backend);

-- User sessions table
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    token_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_token_hash ON user_sessions(token_hash);
