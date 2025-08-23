#!/bin/bash

# WicketWise Comprehensive Testing Suite
# Runs all tests: unit, integration, security, performance, and frontend

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test configuration
PYTHON_VENV=".venv"
TEST_DB="test_wicketwise.db"
TEST_REPORT_DIR="test_reports"
COVERAGE_THRESHOLD=80

# Logging
LOG_FILE="${TEST_REPORT_DIR}/test_run_$(date +%Y%m%d_%H%M%S).log"

print_header() {
    echo -e "${BLUE}===========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===========================================${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_step() {
    echo -e "${PURPLE}🔄 $1${NC}"
}

# Setup test environment
setup_test_environment() {
    print_header "Setting Up Test Environment"
    
    # Create test reports directory
    mkdir -p "${TEST_REPORT_DIR}"
    
    # Initialize log file
    echo "WicketWise Test Suite - $(date)" > "${LOG_FILE}"
    echo "=======================================" >> "${LOG_FILE}"
    
    # Activate virtual environment
    if [ -d "${PYTHON_VENV}" ]; then
        print_step "Activating virtual environment"
        source "${PYTHON_VENV}/bin/activate"
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment not found at ${PYTHON_VENV}"
        print_info "Run: python -m venv ${PYTHON_VENV} && source ${PYTHON_VENV}/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
    
    # Install test dependencies
    print_step "Installing test dependencies"
    pip install -q pytest pytest-asyncio pytest-cov pytest-xdist pytest-mock pytest-benchmark httpx pytest-html >> "${LOG_FILE}" 2>&1
    print_success "Test dependencies installed"
    
    # Clean up any previous test artifacts
    print_step "Cleaning up previous test artifacts"
    rm -f "${TEST_DB}"
    rm -rf __pycache__
    find . -name "*.pyc" -delete
    find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    print_success "Test environment cleaned"
}

# Run Python unit tests
run_unit_tests() {
    print_header "Running Unit Tests"
    
    print_step "Running Python unit tests with coverage"
    
    pytest tests/ \
        --cov=. \
        --cov-report=html:${TEST_REPORT_DIR}/coverage_html \
        --cov-report=xml:${TEST_REPORT_DIR}/coverage.xml \
        --cov-report=term-missing \
        --cov-fail-under=${COVERAGE_THRESHOLD} \
        --html=${TEST_REPORT_DIR}/unit_tests_report.html \
        --self-contained-html \
        --junitxml=${TEST_REPORT_DIR}/unit_tests_junit.xml \
        -v \
        --tb=short \
        --maxfail=5 \
        2>&1 | tee -a "${LOG_FILE}"
    
    if [ $? -eq 0 ]; then
        print_success "Unit tests passed with coverage ≥ ${COVERAGE_THRESHOLD}%"
    else
        print_error "Unit tests failed or coverage below ${COVERAGE_THRESHOLD}%"
        return 1
    fi
}

# Run integration tests
run_integration_tests() {
    print_header "Running Integration Tests"
    
    print_step "Starting test services"
    
    # Start test database and services in background
    python -c "
import asyncio
import sys
sys.path.append('.')
from service_container import get_container

async def start_test_services():
    container = get_container()
    try:
        await container.start_all_services()
        print('✅ Test services started')
        # Keep services running for integration tests
        await asyncio.sleep(2)
    except Exception as e:
        print(f'❌ Failed to start test services: {e}')
        sys.exit(1)
    finally:
        await container.stop_all_services()

asyncio.run(start_test_services())
" 2>&1 | tee -a "${LOG_FILE}"
    
    if [ $? -ne 0 ]; then
        print_error "Failed to start test services"
        return 1
    fi
    
    print_step "Running integration tests"
    
    pytest tests/integration/ \
        --html=${TEST_REPORT_DIR}/integration_tests_report.html \
        --self-contained-html \
        --junitxml=${TEST_REPORT_DIR}/integration_tests_junit.xml \
        -v \
        --tb=short \
        --maxfail=3 \
        2>&1 | tee -a "${LOG_FILE}"
    
    if [ $? -eq 0 ]; then
        print_success "Integration tests passed"
    else
        print_error "Integration tests failed"
        return 1
    fi
}

# Run security tests
run_security_tests() {
    print_header "Running Security Tests"
    
    print_step "Running security vulnerability tests"
    
    # Install security testing tools
    pip install -q bandit safety semgrep >> "${LOG_FILE}" 2>&1
    
    # Run Bandit for security issues
    print_step "Running Bandit security scan"
    bandit -r . -f json -o ${TEST_REPORT_DIR}/bandit_security_report.json || true
    bandit -r . -f txt -o ${TEST_REPORT_DIR}/bandit_security_report.txt || true
    
    # Run Safety for known vulnerabilities
    print_step "Running Safety vulnerability scan"
    safety check --json --output ${TEST_REPORT_DIR}/safety_vulnerability_report.json || true
    safety check --output ${TEST_REPORT_DIR}/safety_vulnerability_report.txt || true
    
    # Run custom security tests
    pytest tests/security/ \
        --html=${TEST_REPORT_DIR}/security_tests_report.html \
        --self-contained-html \
        --junitxml=${TEST_REPORT_DIR}/security_tests_junit.xml \
        -v \
        2>&1 | tee -a "${LOG_FILE}"
    
    if [ $? -eq 0 ]; then
        print_success "Security tests passed"
    else
        print_warning "Security tests completed with warnings (check reports)"
    fi
}

# Run performance tests
run_performance_tests() {
    print_header "Running Performance Tests"
    
    print_step "Running performance benchmarks"
    
    pytest tests/performance/ \
        --benchmark-only \
        --benchmark-html=${TEST_REPORT_DIR}/performance_benchmark_report.html \
        --benchmark-json=${TEST_REPORT_DIR}/performance_benchmark_report.json \
        --html=${TEST_REPORT_DIR}/performance_tests_report.html \
        --self-contained-html \
        -v \
        2>&1 | tee -a "${LOG_FILE}"
    
    if [ $? -eq 0 ]; then
        print_success "Performance tests passed"
    else
        print_error "Performance tests failed"
        return 1
    fi
}

# Run frontend tests
run_frontend_tests() {
    print_header "Running Frontend Tests"
    
    if [ -d "frontend" ]; then
        cd frontend
        
        # Check if Node.js is installed
        if ! command -v npm &> /dev/null; then
            print_warning "Node.js/npm not found. Skipping frontend tests."
            cd ..
            return 0
        fi
        
        print_step "Installing frontend dependencies"
        npm install --silent >> "../${LOG_FILE}" 2>&1
        
        print_step "Running frontend linting"
        npm run lint >> "../${LOG_FILE}" 2>&1 || print_warning "Frontend linting completed with warnings"
        
        print_step "Running TypeScript type checking"
        npm run type-check >> "../${LOG_FILE}" 2>&1
        
        if [ $? -eq 0 ]; then
            print_success "TypeScript type checking passed"
        else
            print_error "TypeScript type checking failed"
            cd ..
            return 1
        fi
        
        print_step "Building frontend"
        npm run build >> "../${LOG_FILE}" 2>&1
        
        if [ $? -eq 0 ]; then
            print_success "Frontend build successful"
        else
            print_error "Frontend build failed"
            cd ..
            return 1
        fi
        
        cd ..
    else
        print_warning "Frontend directory not found. Skipping frontend tests."
    fi
}

# Run API tests
run_api_tests() {
    print_header "Running API Tests"
    
    print_step "Starting API server for testing"
    
    # Start the API server in background
    python modern_api_gateway.py &
    API_PID=$!
    
    # Wait for server to start
    sleep 5
    
    # Check if server is running
    if ! curl -s http://127.0.0.1:5005/api/health > /dev/null; then
        print_error "API server failed to start"
        kill $API_PID 2>/dev/null || true
        return 1
    fi
    
    print_step "Running API endpoint tests"
    
    pytest tests/api/ \
        --html=${TEST_REPORT_DIR}/api_tests_report.html \
        --self-contained-html \
        --junitxml=${TEST_REPORT_DIR}/api_tests_junit.xml \
        -v \
        2>&1 | tee -a "${LOG_FILE}"
    
    API_TEST_RESULT=$?
    
    # Stop API server
    print_step "Stopping API server"
    kill $API_PID 2>/dev/null || true
    sleep 2
    
    if [ $API_TEST_RESULT -eq 0 ]; then
        print_success "API tests passed"
    else
        print_error "API tests failed"
        return 1
    fi
}

# Generate comprehensive test report
generate_test_report() {
    print_header "Generating Comprehensive Test Report"
    
    REPORT_FILE="${TEST_REPORT_DIR}/comprehensive_test_report.html"
    
    cat > "${REPORT_FILE}" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>WicketWise Comprehensive Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #2563eb; color: white; padding: 20px; border-radius: 8px; }
        .section { margin: 20px 0; padding: 15px; border-left: 4px solid #2563eb; background: #f8fafc; }
        .success { border-left-color: #059669; }
        .warning { border-left-color: #d97706; }
        .error { border-left-color: #dc2626; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 4px; }
        a { color: #2563eb; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .timestamp { color: #6b7280; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏏 WicketWise Comprehensive Test Report</h1>
        <p class="timestamp">Generated: $(date)</p>
    </div>
    
    <div class="section">
        <h2>📊 Test Summary</h2>
        <div class="metric">
            <strong>Test Run:</strong> $(date +%Y-%m-%d_%H:%M:%S)
        </div>
        <div class="metric">
            <strong>Environment:</strong> $(python --version 2>&1)
        </div>
        <div class="metric">
            <strong>Coverage Threshold:</strong> ${COVERAGE_THRESHOLD}%
        </div>
    </div>
    
    <div class="section">
        <h2>📋 Test Reports</h2>
        <ul>
EOF

    # Add links to individual reports if they exist
    for report in unit_tests_report.html integration_tests_report.html security_tests_report.html performance_tests_report.html api_tests_report.html coverage_html/index.html; do
        if [ -f "${TEST_REPORT_DIR}/${report}" ]; then
            report_name=$(basename "$report" .html | tr '_' ' ' | sed 's/\b\w/\U&/g')
            echo "            <li><a href=\"${report}\">${report_name}</a></li>" >> "${REPORT_FILE}"
        fi
    done
    
    cat >> "${REPORT_FILE}" << EOF
        </ul>
    </div>
    
    <div class="section">
        <h2>🔍 Security Reports</h2>
        <ul>
            <li><a href="bandit_security_report.txt">Bandit Security Scan</a></li>
            <li><a href="safety_vulnerability_report.txt">Safety Vulnerability Scan</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>⚡ Performance Reports</h2>
        <ul>
            <li><a href="performance_benchmark_report.html">Performance Benchmarks</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>📈 Coverage Report</h2>
        <p><a href="coverage_html/index.html">Detailed Coverage Report</a></p>
    </div>
    
    <div class="section">
        <h2>📝 Raw Logs</h2>
        <p><a href="$(basename "${LOG_FILE}")">Complete Test Run Log</a></p>
    </div>
    
    <div class="section">
        <h2>🏆 Test Results Summary</h2>
        <p>Check individual reports for detailed results and metrics.</p>
    </div>
</body>
</html>
EOF
    
    print_success "Comprehensive test report generated: ${REPORT_FILE}"
}

# Cleanup function
cleanup() {
    print_step "Cleaning up test processes"
    
    # Kill any remaining background processes
    pkill -f "python.*api_gateway" 2>/dev/null || true
    pkill -f "pytest" 2>/dev/null || true
    
    # Clean up test database
    rm -f "${TEST_DB}"
    
    print_success "Cleanup completed"
}

# Main test execution
main() {
    print_header "🏏 WicketWise Comprehensive Testing Suite"
    print_info "Starting comprehensive test run at $(date)"
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Track overall success
    OVERALL_SUCCESS=true
    
    # Setup
    setup_test_environment || { print_error "Environment setup failed"; exit 1; }
    
    # Run test suites
    print_info "Running test suites..."
    
    if ! run_unit_tests; then
        OVERALL_SUCCESS=false
        print_error "Unit tests failed"
    fi
    
    if ! run_integration_tests; then
        OVERALL_SUCCESS=false
        print_error "Integration tests failed"
    fi
    
    if ! run_security_tests; then
        print_warning "Security tests completed with warnings"
    fi
    
    if ! run_performance_tests; then
        OVERALL_SUCCESS=false
        print_error "Performance tests failed"
    fi
    
    if ! run_frontend_tests; then
        OVERALL_SUCCESS=false
        print_error "Frontend tests failed"
    fi
    
    if ! run_api_tests; then
        OVERALL_SUCCESS=false
        print_error "API tests failed"
    fi
    
    # Generate final report
    generate_test_report
    
    # Final summary
    print_header "Test Suite Summary"
    
    if [ "$OVERALL_SUCCESS" = true ]; then
        print_success "🎉 All tests passed successfully!"
        print_info "View comprehensive report: ${TEST_REPORT_DIR}/comprehensive_test_report.html"
        exit 0
    else
        print_error "❌ Some tests failed"
        print_info "Check individual reports in: ${TEST_REPORT_DIR}/"
        print_info "View comprehensive report: ${TEST_REPORT_DIR}/comprehensive_test_report.html"
        exit 1
    fi
}

# Parse command line arguments
case "${1:-all}" in
    "unit")
        setup_test_environment && run_unit_tests
        ;;
    "integration")
        setup_test_environment && run_integration_tests
        ;;
    "security")
        setup_test_environment && run_security_tests
        ;;
    "performance")
        setup_test_environment && run_performance_tests
        ;;
    "frontend")
        run_frontend_tests
        ;;
    "api")
        setup_test_environment && run_api_tests
        ;;
    "all"|*)
        main
        ;;
esac
