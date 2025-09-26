# RESTful API & Flask Assignment Solutions
# Complete solutions for all theoretical and practical questions

"""
INSTALLATION REQUIREMENTS:
pip install flask
pip install flask-sqlalchemy
pip install flask-restful
pip install wtforms
pip install flask-wtf
"""

from flask import Flask, request, jsonify, render_template_string, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_restful import Api, Resource
from werkzeug.exceptions import NotFound, BadRequest
import os
from datetime import datetime
import json

print("Flask and related libraries imported successfully!")

# ============================================================================
# THEORETICAL QUESTIONS - ANSWERS
# ============================================================================

print("\n" + "="*80)
print("THEORETICAL QUESTIONS - ANSWERS")
print("="*80)

print("\n1. What is a RESTful API?")
print("""
RESTful API (Representational State Transfer) is an architectural style for building web services.

Key Principles:
- Client-Server Architecture: Separation of concerns
- Stateless: Each request must contain all information needed
- Cacheable: Responses should be cacheable when appropriate
- Uniform Interface: Consistent way to interact with resources
- Layered System: Architecture can be composed of hierarchical layers
- Code on Demand (optional): Server can extend client functionality

REST uses standard HTTP methods:
- GET: Retrieve data
- POST: Create new resource
- PUT: Update/replace entire resource
- PATCH: Partial update of resource
- DELETE: Remove resource

URLs represent resources, and HTTP methods represent actions on those resources.
""")

print("\n2. Explain the concept of API specification")
print("""
API Specification is a detailed description of how an API works.

Components:
- Endpoints: Available URLs and their purposes
- HTTP Methods: Which methods each endpoint supports
- Request/Response Format: Expected data structure
- Authentication: How to authenticate requests
- Error Handling: Error codes and messages
- Rate Limiting: Usage restrictions

Popular specification formats:
- OpenAPI (Swagger): Most widely used, JSON/YAML format
- RAML: RESTful API Modeling Language
- API Blueprint: Markdown-based format

Benefits:
- Clear communication between teams
- Automatic documentation generation
- Code generation for clients and servers
- Testing and validation tools
- Better API design and consistency
""")

print("\n3. What is Flask, and why is it popular for building APIs?")
print("""
Flask is a lightweight, flexible Python web framework.

Key Features:
- Minimalist: Small core with extensions
- Flexible: No rigid structure imposed
- WSGI compliant: Works with various web servers
- Built-in development server and debugger
- Jinja2 templating engine
- Secure cookie handling (sessions)

Why popular for APIs:
- Quick to set up and deploy
- Lightweight and fast
- Excellent for microservices
- Great ecosystem of extensions
- Easy to understand and learn
- RESTful request dispatching
- JSON support built-in
- Flexible routing system
""")

print("\n4. What is routing in Flask?")
print("""
Routing in Flask is the mechanism that maps URLs to Python functions.

Key Concepts:
- URL patterns are mapped to view functions
- Routes can include variable parts
- Different HTTP methods can be specified
- URL parameters can be captured and typed
- Routes can be organized using Blueprints

Route Examples:
- @app.route('/') - Root URL
- @app.route('/user/<username>') - Variable route
- @app.route('/post/<int:id>') - Typed variable
- @app.route('/api/data', methods=['GET', 'POST']) - Multiple methods

Flask uses Werkzeug's routing system which is efficient and flexible.
""")

print("\n5. How do you create a simple Flask application?")
print("""
Basic Flask Application Structure:

1. Import Flask
2. Create Flask instance
3. Define routes with @app.route decorator
4. Create view functions
5. Run the application

Minimum example:
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
""")

print("\n6. What are HTTP methods used in RESTful APIs?")
print("""
Standard HTTP methods in RESTful APIs:

GET: Retrieve data
- Safe and idempotent
- Should not modify server state
- Used for reading resources

POST: Create new resources
- Not idempotent
- Sends data in request body
- Used for creating new resources

PUT: Update/replace entire resource
- Idempotent
- Replaces entire resource
- Used for complete updates

PATCH: Partial update
- Not necessarily idempotent
- Updates specific fields
- Used for partial modifications

DELETE: Remove resource
- Idempotent
- Removes specified resource
- Used for deletion operations

HEAD: Get headers only (like GET but no body)
OPTIONS: Get allowed methods for a resource
""")

print("\n7. What is the purpose of the @app.route() decorator in Flask?")
print("""
The @app.route() decorator binds a URL pattern to a view function.

Purpose:
- Maps URLs to Python functions
- Defines which HTTP methods are allowed
- Captures URL parameters
- Enables URL generation
- Supports URL building with url_for()

Parameters:
- rule: URL pattern (required)
- methods: List of allowed HTTP methods
- defaults: Default values for variables
- subdomain: Subdomain matching
- strict_slashes: Trailing slash behavior
- redirect_to: Redirect to another endpoint

The decorator registers the function with Flask's routing system.
""")

print("\n8. What is the difference between GET and POST HTTP methods?")
print("""
GET vs POST Methods:

GET:
- Retrieves data from server
- Parameters in URL query string
- Idempotent and safe
- Can be cached by browsers
- Limited data size (URL length limits)
- Visible in browser history/logs
- Should not modify server state

POST:
- Sends data to server
- Data in request body
- Not idempotent
- Not cached by default
- Large data size allowed
- Data not visible in URL
- Can modify server state
- Used for creating resources

Security: POST is more secure for sensitive data as parameters aren't in URL.
""")

print("\n9. How do you handle errors in Flask APIs?")
print("""
Error Handling in Flask APIs:

1. Built-in Error Handlers:
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

2. Custom Exceptions:
class ValidationError(Exception):
    pass

@app.errorhandler(ValidationError)
def handle_validation_error(e):
    return jsonify({'error': str(e)}), 400

3. Try-Catch in Routes:
try:
    # API logic
except Exception as e:
    return jsonify({'error': 'Internal server error'}), 500

4. Flask-RESTful Error Handling:
Built-in error handling for common HTTP errors

Best Practices:
- Return consistent JSON error format
- Use appropriate HTTP status codes
- Log errors for debugging
- Don't expose sensitive information
""")

print("\n10. How do you connect Flask to a SQL database?")
print("""
Connecting Flask to SQL Database:

1. Using Flask-SQLAlchemy (Recommended):
from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
db = SQLAlchemy(app)

2. Direct database connections:
import sqlite3
def get_db_connection():
    conn = sqlite3.connect('database.db')
    return conn

3. Configuration:
- Set database URI in config
- Handle connection pooling
- Manage transactions
- Error handling

Supported databases:
- SQLite (development)
- PostgreSQL (production)
- MySQL
- Oracle
- SQL Server
""")

# Continue with remaining theoretical questions...
print("\n11. What is the role of Flask-SQLAlchemy?")
print("""
Flask-SQLAlchemy is an extension that adds SQLAlchemy support to Flask.

Key Features:
- ORM (Object-Relational Mapping)
- Database abstraction layer
- Query builder
- Relationship management
- Migration support (with Flask-Migrate)
- Connection pooling
- Transaction management

Benefits:
- Pythonic database operations
- Database-agnostic code
- Automatic SQL generation
- Model validation
- Lazy loading of relationships
- Built-in pagination
- Integration with Flask app context
""")

print("\n12. What are Flask blueprints, and how are they useful?")
print("""
Flask Blueprints are a way to organize Flask applications into components.

Purpose:
- Modular application structure
- Reusable application components
- Better code organization
- Team collaboration
- Plugin-like architecture

Benefits:
- Group related routes together
- Apply common decorators to route groups
- Register error handlers for specific modules
- Serve static files from different locations
- Create reusable application components

Use Cases:
- API versioning (/api/v1/, /api/v2/)
- Feature-based organization
- Admin panels
- User authentication modules
""")

# ============================================================================
# PRACTICAL QUESTIONS - CODE SOLUTIONS
# ============================================================================

print("\n" + "="*80)
print("PRACTICAL QUESTIONS - CODE SOLUTIONS")
print("="*80)

# Create a comprehensive Flask application with all examples
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for sessions
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///example.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
api = Api(app)

print("\n1. How do you create a basic Flask application?")
basic_app_code = '''
from flask import Flask

# Create Flask instance
app = Flask(__name__)

# Define a simple route
@app.route('/')
def home():
    return 'Hello, World! This is a basic Flask application.'

@app.route('/about')
def about():
    return 'This is the about page.'

# Run the application
if __name__ == '__main__':
    app.run(debug=True, port=5000)
'''
print("Basic Flask Application:")
print(basic_app_code)

print("\n2. How do you serve static files like images or CSS in Flask?")
# Static files example
@app.route('/static-example')
def static_example():
    return '''
    <html>
    <head>
        <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    </head>
    <body>
        <h1>Static Files Example</h1>
        <img src="{{ url_for('static', filename='images/logo.png') }}" alt="Logo">
        <script src="{{ url_for('static', filename='script.js') }}"></script>
    </body>
    </html>
    '''

static_files_code = '''
# Flask automatically serves files from the 'static' folder
# Directory structure:
# /project
#   /static
#     /css
#       style.css
#     /js
#       script.js
#     /images
#       logo.png
#   app.py

# In templates, use url_for to generate URLs:
# <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
# <img src="{{ url_for('static', filename='images/logo.png') }}">
'''
print("Static Files Serving:")
print(static_files_code)

print("\n3. How do you define different routes with different HTTP methods?")
# Different HTTP methods example
@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify({
        'users': [
            {'id': 1, 'name': 'John Doe', 'email': 'john@example.com'},
            {'id': 2, 'name': 'Jane Smith', 'email': 'jane@example.com'}
        ]
    })

@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.get_json()
    if not data or 'name' not in data or 'email' not in data:
        return jsonify({'error': 'Name and email are required'}), 400
    
    # Simulate user creation
    new_user = {
        'id': 3,  # In real app, this would be generated
        'name': data['name'],
        'email': data['email']
    }
    return jsonify(new_user), 201

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    # Simulate database lookup
    user = {'id': user_id, 'name': 'User Name', 'email': 'user@example.com'}
    return jsonify(user)

@app.route('/api/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    data = request.get_json()
    # Simulate user update
    updated_user = {
        'id': user_id,
        'name': data.get('name', 'Updated Name'),
        'email': data.get('email', 'updated@example.com')
    }
    return jsonify(updated_user)

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    # Simulate user deletion
    return jsonify({'message': f'User {user_id} deleted successfully'}), 200

http_methods_code = '''
# Different HTTP methods for the same route
@app.route('/api/resource', methods=['GET', 'POST'])
def handle_resource():
    if request.method == 'GET':
        return jsonify({'action': 'retrieve'})
    elif request.method == 'POST':
        return jsonify({'action': 'create'})

# Separate functions for different methods
@app.route('/api/items', methods=['GET'])
def get_items():
    return jsonify({'items': []})

@app.route('/api/items', methods=['POST'])
def create_item():
    data = request.get_json()
    return jsonify({'created': data}), 201

# RESTful routing with parameters
@app.route('/api/items/<int:item_id>', methods=['GET', 'PUT', 'DELETE'])
def handle_item(item_id):
    if request.method == 'GET':
        return jsonify({'id': item_id, 'name': 'Item'})
    elif request.method == 'PUT':
        return jsonify({'id': item_id, 'updated': True})
    elif request.method == 'DELETE':
        return jsonify({'id': item_id, 'deleted': True})
'''
print("HTTP Methods Example:")
print(http_methods_code)

print("\n4. How do you render HTML templates in Flask?")
# HTML template rendering
@app.route('/template-example')
def template_example():
    user = {'name': 'John Doe', 'email': 'john@example.com'}
    items = ['Item 1', 'Item 2', 'Item 3']
    
    template = '''
    <html>
    <head><title>Template Example</title></head>
    <body>
        <h1>Welcome, {{ user.name }}!</h1>
        <p>Email: {{ user.email }}</p>
        <h2>Your Items:</h2>
        <ul>
        {% for item in items %}
            <li>{{ item }}</li>
        {% endfor %}
        </ul>
        <p>Current time: {{ current_time }}</p>
    </body>
    </html>
    '''
    
    return render_template_string(template, 
                                user=user, 
                                items=items, 
                                current_time=datetime.now())

template_code = '''
# Template rendering in Flask
from flask import render_template

@app.route('/profile/<username>')
def profile(username):
    user_data = {
        'username': username,
        'email': f'{username}@example.com',
        'posts': ['Post 1', 'Post 2']
    }
    # render_template looks for templates in 'templates' folder
    return render_template('profile.html', user=user_data)

# templates/profile.html:
<!DOCTYPE html>
<html>
<head>
    <title>{{ user.username }} Profile</title>
</head>
<body>
    <h1>{{ user.username }}'s Profile</h1>
    <p>Email: {{ user.email }}</p>
    
    <h2>Posts:</h2>
    <ul>
    {% for post in user.posts %}
        <li>{{ post }}</li>
    {% endfor %}
    </ul>
    
    {% if user.posts %}
        <p>User has {{ user.posts|length }} posts.</p>
    {% else %}
        <p>No posts yet.</p>
    {% endif %}
</body>
</html>
'''
print("Template Rendering:")
print(template_code)

print("\n5. How can you generate URLs for routes in Flask using url_for?")
@app.route('/url-examples')
def url_examples():
    examples = {
        'home_url': url_for('home'),
        'user_profile': url_for('get_user', user_id=123),
        'static_css': url_for('static', filename='style.css'),
        'external_url': url_for('url_examples', _external=True)
    }
    return jsonify(examples)

url_for_code = '''
from flask import url_for

# Generate URL for a route
@app.route('/')
def home():
    return 'Home Page'

@app.route('/user/<int:user_id>')
def user_profile(user_id):
    return f'User {user_id} Profile'

# Using url_for in views
@app.route('/navigation')
def navigation():
    urls = {
        'home': url_for('home'),                           # '/'
        'user': url_for('user_profile', user_id=123),     # '/user/123'
        'static': url_for('static', filename='style.css'), # '/static/style.css'
        'external': url_for('home', _external=True)        # 'http://localhost:5000/'
    }
    return jsonify(urls)

# In templates:
# <a href="{{ url_for('home') }}">Home</a>
# <a href="{{ url_for('user_profile', user_id=user.id) }}">Profile</a>
# <img src="{{ url_for('static', filename='logo.png') }}">
'''
print("URL Generation:")
print(url_for_code)

print("\n6. How do you handle forms in Flask?")
@app.route('/form-example', methods=['GET', 'POST'])
def form_example():
    if request.method == 'GET':
        # Show form
        form_html = '''
        <form method="POST">
            <label>Name: <input type="text" name="name" required></label><br><br>
            <label>Email: <input type="email" name="email" required></label><br><br>
            <label>Age: <input type="number" name="age" min="1" max="120"></label><br><br>
            <label>Message: <textarea name="message"></textarea></label><br><br>
            <input type="submit" value="Submit">
        </form>
        '''
        return form_html
    
    elif request.method == 'POST':
        # Process form data
        name = request.form.get('name')
        email = request.form.get('email')
        age = request.form.get('age', type=int)
        message = request.form.get('message')
        
        # Validate data
        if not name or not email:
            return jsonify({'error': 'Name and email are required'}), 400
        
        # Process the data (save to database, send email, etc.)
        result = {
            'message': 'Form submitted successfully!',
            'data': {
                'name': name,
                'email': email,
                'age': age,
                'message': message
            }
        }
        return jsonify(result)

form_handling_code = '''
from flask import request, render_template, flash, redirect, url_for

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        # Validate form data
        if not name or not email:
            flash('Name and email are required!')
            return redirect(url_for('contact'))
        
        # Process form (save to database, send email, etc.)
        # ... processing logic ...
        
        flash('Thank you for your message!')
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

# File upload handling
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'message': 'File uploaded successfully'})
'''
print("Form Handling:")
print(form_handling_code)

print("\n7. How can you validate form data in Flask?")
form_validation_code = '''
from flask import request, jsonify
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    pattern = r'^\+?1?[-.\s]?\\(?[0-9]{3}\\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
    return re.match(pattern, phone) is not None

@app.route('/validate-form', methods=['POST'])
def validate_form():
    errors = []
    
    # Get form data
    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip()
    age = request.form.get('age')
    phone = request.form.get('phone', '').strip()
    
    # Validate name
    if not name:
        errors.append('Name is required')
    elif len(name) < 2:
        errors.append('Name must be at least 2 characters')
    
    # Validate email
    if not email:
        errors.append('Email is required')
    elif not validate_email(email):
        errors.append('Invalid email format')
    
    # Validate age
    if age:
        try:
            age = int(age)
            if age < 0 or age > 120:
                errors.append('Age must be between 0 and 120')
        except ValueError:
            errors.append('Age must be a number')
    
    # Validate phone (optional)
    if phone and not validate_phone(phone):
        errors.append('Invalid phone number format')
    
    if errors:
        return jsonify({'errors': errors}), 400
    
    return jsonify({'message': 'Form is valid', 'data': {
        'name': name,
        'email': email,
        'age': age,
        'phone': phone
    }})

# Using Flask-WTF for advanced validation
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, TextAreaField
from wtforms.validators import DataRequired, Email, NumberRange, Length

class ContactForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    age = IntegerField('Age', validators=[NumberRange(min=0, max=120)])
    message = TextAreaField('Message', validators=[Length(max=500)])

@app.route('/wtf-form', methods=['GET', 'POST'])
def wtf_form():
    form = ContactForm()
    if form.validate_on_submit():
        # Form is valid, process data
        return jsonify({'message': 'Form submitted successfully!'})
    else:
        # Form has errors
        return jsonify({'errors': form.errors}), 400
'''
print("Form Validation:")
print(form_validation_code)

print("\n8. How do you manage sessions in Flask?")
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')
    
    # Simulate authentication
    if username == 'admin' and password == 'password':
        session['user_id'] = 1
        session['username'] = username
        session['logged_in'] = True
        return jsonify({'message': 'Login successful'})
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/logout')
def logout():
    session.clear()  # Clear all session data
    return jsonify({'message': 'Logged out successfully'})

@app.route('/profile')
def profile():
    if 'logged_in' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    return jsonify({
        'user_id': session.get('user_id'),
        'username': session.get('username'),
        'logged_in': session.get('logged_in')
    })

session_code = '''
from flask import session

# Configure secret key for sessions
app.secret_key = 'your-secret-key'

# Store data in session
@app.route('/set-session')
def set_session():
    session['username'] = 'john_doe'
    session['user_id'] = 123
    session['preferences'] = {'theme': 'dark', 'language': 'en'}
    return 'Session data set'

# Retrieve data from session
@app.route('/get-session')
def get_session():
    username = session.get('username')
    user_id = session.get('user_id')
    preferences = session.get('preferences', {})
    
    if username:
        return f'Welcome {username}! ID: {user_id}'
    else:
        return 'No session data found'

# Check if user is logged in
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Login required'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/protected')
@login_required
def protected():
    return 'This is a protected route'

# Clear session
@app.route('/clear-session')
def clear_session():
    session.clear()
    return 'Session cleared'

# Session configuration options
app.config['SESSION_COOKIE_SECURE'] = True      # HTTPS only
app.config['SESSION_COOKIE_HTTPONLY'] = True    # No JavaScript access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'   # CSRF protection
'''
print("Session Management:")
print(session_code)

print("\n9. How do you redirect to a different route in Flask?")
@app.route('/redirect-example')
def redirect_example():
    # Redirect to another route
    return redirect(url_for('home'))

@app.route('/conditional-redirect')
def conditional_redirect():
    # Conditional redirect based on user status
    if 'logged_in' in session:
        return redirect(url_for('profile'))
    else:
        return redirect(url_for('login'))

@app.route('/redirect-with-params')
def redirect_with_params():
    # Redirect with parameters
    return redirect(url_for('get_user', user_id=123))

redirect_code = '''
from flask import redirect, url_for, request

# Simple redirect
@app.route('/old-page')
def old_page():
    return redirect(url_for('new_page'))

@app.route('/new-page')
def new_page():
    return 'This is the new page'

# Redirect with status code
@app.route('/moved')
def moved():
    return redirect(url_for('new_location'), code=301)  # Permanent redirect

# Conditional redirect
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return 'Dashboard content'

# Redirect to external URL
@app.route('/external')
def external():
    return redirect('https://www.example.com')

# Redirect after form submission
@app.route('/submit', methods=['POST'])
def submit():
    # Process form data
    flash('Form submitted successfully!')
    return redirect(url_for('form_page'))

# Redirect with query parameters
@app.route('/search-redirect')
def search_redirect():
    query = request.args.get('q', '')
    return redirect(url_for('search_results', query=query))

# Redirect to previous page
@app.route('/go-back')
def go_back():
    return redirect(request.referrer or url_for('home'))
'''
print("Redirects:")
print(redirect_code)

print("\n10. How do you handle errors in Flask (e.g., 404)?")
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(400)
def bad_request_error(error):
    return jsonify({'error': 'Bad request'}), 400

# Custom error route to demonstrate
@app.route('/trigger-error/<error_type>')
def trigger_error(error_type):
    if error_type == '404':
        return not_found_error(None)
    elif error_type == '500':
        raise Exception("Simulated server error")
    elif error_type == '400':
        return bad_request_error(None)
    else:
        return jsonify({'message': 'No error triggered'})

error_handling_code = '''
from flask import jsonify
from werkzeug.exceptions import NotFound, BadRequest

# Built-in error handlers
@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(403)
def forbidden(e):
    return jsonify({'error': 'Access forbidden'}), 403

# Custom exception handling
class ValidationError(Exception):
    pass

@app.errorhandler(ValidationError)
def handle_validation_error(e):
    return jsonify({'error': str(e)}), 400

# Global error handler for all HTTP exceptions
from werkzeug.exceptions import HTTPException

@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    app.logger.error(f'Unhandled exception: {e}')
    
    if isinstance(e, HTTPException):
        return jsonify({'error': e.description}), e.code
    
    # Return 500 for any other exception
    return jsonify({'error': 'Internal server error'}), 500

# Try-catch in routes
@app.route('/api/risky-operation')
def risky_operation():
    try:
        # Some operation that might fail
        result = 10 / 0  # This will raise ZeroDivisionError
        return jsonify({'result': result})
    except ZeroDivisionError:
        return jsonify({'error': 'Division by zero'}), 400
    except Exception as e:
        return jsonify({'error': 'Unexpected error occurred'}), 500

# Error handling with logging
import logging

@app.route('/api/with-logging')
def with_logging():
    try:
        # Simulate some operation
        app.logger.info('Starting operation')
        # ... operation code ...
        app.logger.info('Operation completed successfully')
        return jsonify({'status': 'success'})
    except Exception as e:
        app.logger.error(f'Operation failed: {str(e)}')
        return jsonify({'error': 'Operation failed'}), 500
'''
print("Error Handling:")
print(error_handling_code)

print("\n11. How do you structure a Flask app using Blueprints?")

# Blueprint example
from flask import Blueprint

# Create blueprints
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# API Blueprint routes
@api_bp.route('/users')
def api_users():
    return jsonify({'users': ['user1', 'user2', 'user3']})

@api_bp.route('/products')
def api_products():
    return jsonify({'products': ['product1', 'product2']})

# Auth Blueprint routes
@auth_bp.route('/login', methods=['POST'])
def auth_login():
    return jsonify({'message': 'Login endpoint'})

@auth_bp.route('/register', methods=['POST'])
def auth_register():
    return jsonify({'message': 'Register endpoint'})

# Admin Blueprint routes
@admin_bp.route('/dashboard')
def admin_dashboard():
    return jsonify({'message': 'Admin dashboard'})

@admin_bp.route('/users')
def admin_users():
    return jsonify({'message': 'Admin users management'})

# Register blueprints
app.register_blueprint(api_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)

blueprint_code = '''
# Project structure with blueprints:
# /project
#   /app
#     __init__.py
#     /auth
#       __init__.py
#       routes.py
#     /api
#       __init__.py
#       routes.py
#     /admin
#       __init__.py
#       routes.py
#   run.py

# app/__init__.py
from flask import Flask
from app.auth import auth_bp
from app.api import api_bp
from app.admin import admin_bp

def create_app():
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(admin_bp)
    
    return app

# app/auth/__init__.py
from flask import Blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')
from . import routes

# app/auth/routes.py
from flask import jsonify, request
from . import auth_bp

@auth_bp.route('/login', methods=['POST'])
def login():
    return jsonify({'message': 'Login successful'})

@auth_bp.route('/logout')
def logout():
    return jsonify({'message': 'Logged out'})

# app/api/__init__.py
from flask import Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')
from . import routes

# app/api/routes.py
from flask import jsonify
from . import api_bp

@api_bp.route('/users')
def get_users():
    return jsonify({'users': []})

@api_bp.route('/products')
def get_products():
    return jsonify({'products': []})

# Blueprint with error handlers
@api_bp.errorhandler(404)
def api_not_found(error):
    return jsonify({'error': 'API endpoint not found'}), 404

# Blueprint with before_request
@api_bp.before_request
def before_api_request():
    # Authentication check for all API routes
    pass
'''
print("Blueprints Structure:")
print(blueprint_code)

print("\n12. How do you define a custom Jinja filter in Flask?")

# Custom Jinja filters
@app.template_filter('reverse')
def reverse_filter(s):
    """Reverse a string"""
    return s[::-1]

@app.template_filter('currency')
def currency_filter(amount):
    """Format number as currency"""
    return f"${amount:,.2f}"

@app.template_filter('truncate_words')
def truncate_words_filter(text, length=10):
    """Truncate text to specified number of words"""
    words = text.split()
    if len(words) <= length:
        return text
    return ' '.join(words[:length]) + '...'

@app.route('/custom-filters-demo')
def custom_filters_demo():
    template = '''
    <h1>Custom Jinja Filters Demo</h1>
    <p>Original: {{ text }}</p>
    <p>Reversed: {{ text|reverse }}</p>
    <p>Price: {{ price|currency }}</p>
    <p>Truncated: {{ long_text|truncate_words(5) }}</p>
    '''
    
    return render_template_string(template,
                                text="Hello World",
                                price=1234.56,
                                long_text="This is a very long text that needs to be truncated for display purposes")

jinja_filters_code = '''
# Custom Jinja filters in Flask

# Method 1: Using decorator
@app.template_filter('reverse')
def reverse_filter(s):
    return s[::-1]

@app.template_filter('currency')
def currency_filter(amount):
    return f"${amount:,.2f}"

# Method 2: Manual registration
def format_datetime(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

app.jinja_env.filters['datetime'] = format_datetime

# Advanced filter with parameters
@app.template_filter('truncate_words')
def truncate_words(text, length=10, suffix='...'):
    words = text.split()
    if len(words) <= length:
        return text
    return ' '.join(words[:length]) + suffix

# Filter for highlighting search terms
@app.template_filter('highlight')
def highlight_filter(text, search_term):
    if not search_term:
        return text
    highlighted = text.replace(
        search_term, 
        f'<mark>{search_term}</mark>'
    )
    return highlighted

# Usage in templates:
# {{ "Hello World"|reverse }}  -> "dlroW olleH"
# {{ 1234.56|currency }}       -> "$1,234.56"
# {{ timestamp|datetime }}      -> "2023-01-15 14:30:00"
# {{ long_text|truncate_words(5) }}
# {{ content|highlight(search_term)|safe }}

# Custom filter for JSON pretty printing
import json
@app.template_filter('tojson_pretty')
def to_json_pretty(obj):
    return json.dumps(obj, indent=2, sort_keys=True)
'''
print("Custom Jinja Filters:")
print(jinja_filters_code)

print("\n13. How can you redirect with query parameters in Flask?")

@app.route('/search-redirect')
def search_redirect():
    # Redirect with query parameters
    query = request.args.get('q', 'default')
    category = request.args.get('category', 'all')
    
    # Build URL with query parameters
    from urllib.parse import urlencode
    params = {'query': query, 'category': category, 'page': 1}
    query_string = urlencode(params)
    
    return redirect(f"/search-results?{query_string}")

@app.route('/search-results')
def search_results():
    query = request.args.get('query', '')
    category = request.args.get('category', 'all')
    page = request.args.get('page', 1, type=int)
    
    return jsonify({
        'query': query,
        'category': category,
        'page': page,
        'results': f'Search results for "{query}" in category "{category}" (page {page})'
    })

query_params_code = '''
from flask import redirect, url_for, request
from urllib.parse import urlencode

# Redirect with query parameters - Method 1
@app.route('/redirect-with-params')
def redirect_with_params():
    params = {'q': 'python', 'category': 'programming', 'sort': 'date'}
    query_string = urlencode(params)
    return redirect(f'/search?{query_string}')

# Redirect with query parameters - Method 2 (using url_for)
@app.route('/redirect-url-for')
def redirect_url_for():
    return redirect(url_for('search', q='python', category='programming'))

@app.route('/search')
def search():
    query = request.args.get('q', '')
    category = request.args.get('category', 'all')
    return f'Searching for {query} in {category}'

# Preserve query parameters during redirect
@app.route('/login-redirect')
def login_redirect():
    # Get the original URL with parameters
    next_url = request.url
    return redirect(url_for('login', next=next_url))

@app.route('/login')
def login():
    next_url = request.args.get('next')
    if next_url:
        # After login, redirect back to original URL
        return redirect(next_url)
    return 'Login page'

# Conditional redirect with parameters
@app.route('/process-form', methods=['POST'])
def process_form():
    success = True  # Simulate form processing
    
    if success:
        return redirect(url_for('success', message='Form submitted successfully'))
    else:
        return redirect(url_for('error', error='Form validation failed'))

@app.route('/success')
def success():
    message = request.args.get('message', 'Success!')
    return f'Success: {message}'

@app.route('/error')
def error():
    error_msg = request.args.get('error', 'An error occurred')
    return f'Error: {error_msg}'
'''
print("Query Parameters in Redirects:")
print(query_params_code)

print("\n14. How do you return JSON responses in Flask?")

@app.route('/json-examples/simple')
def simple_json():
    return jsonify({'message': 'Hello, World!', 'status': 'success'})

@app.route('/json-examples/complex')
def complex_json():
    data = {
        'users': [
            {'id': 1, 'name': 'John Doe', 'email': 'john@example.com', 'active': True},
            {'id': 2, 'name': 'Jane Smith', 'email': 'jane@example.com', 'active': False}
        ],
        'pagination': {
            'page': 1,
            'per_page': 10,
            'total': 2,
            'pages': 1
        },
        'timestamp': datetime.now().isoformat(),
        'metadata': {
            'version': '1.0',
            'server': 'Flask'
        }
    }
    return jsonify(data)

@app.route('/json-examples/with-status')
def json_with_status():
    return jsonify({'error': 'Resource not found'}), 404

@app.route('/json-examples/custom-headers')
def json_with_headers():
    response = jsonify({'data': 'Custom headers example'})
    response.headers['X-Custom-Header'] = 'Custom Value'
    response.headers['X-API-Version'] = '1.0'
    return response

json_responses_code = '''
from flask import jsonify, make_response
import json
from datetime import datetime

# Simple JSON response
@app.route('/api/simple')
def simple_json():
    return jsonify({'message': 'Hello, World!'})

# JSON with custom status code
@app.route('/api/error')
def json_error():
    return jsonify({'error': 'Not found'}), 404

# JSON with custom headers
@app.route('/api/with-headers')
def json_with_headers():
    response = jsonify({'data': 'example'})
    response.headers['X-Total-Count'] = '100'
    response.headers['X-Rate-Limit'] = '1000'
    return response

# Complex JSON structure
@app.route('/api/complex')
def complex_json():
    data = {
        'users': [
            {'id': 1, 'name': 'John', 'email': 'john@example.com'},
            {'id': 2, 'name': 'Jane', 'email': 'jane@example.com'}
        ],
        'meta': {
            'total': 2,
            'page': 1,
            'timestamp': datetime.utcnow().isoformat()
        }
    }
    return jsonify(data)

# Using make_response for more control
@app.route('/api/custom-response')
def custom_response():
    data = {'message': 'Custom response'}
    response = make_response(jsonify(data))
    response.status_code = 201
    response.headers['Location'] = '/api/resource/123'
    return response

# JSON with JSONP callback (for cross-domain requests)
@app.route('/api/jsonp')
def jsonp_response():
    callback = request.args.get('callback', 'callback')
    data = {'message': 'JSONP response'}
    json_data = json.dumps(data)
    return f'{callback}({json_data})', 200, {'Content-Type': 'application/javascript'}

# Handling JSON in POST requests
@app.route('/api/post-json', methods=['POST'])
def handle_json_post():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    
    # Process the JSON data
    processed_data = {
        'received': data,
        'processed_at': datetime.utcnow().isoformat(),
        'status': 'success'
    }
    
    return jsonify(processed_data), 201

# Error handling with JSON
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
'''
print("JSON Responses:")
print(json_responses_code)

print("\n15. How do you capture URL parameters in Flask?")

# URL parameters examples
@app.route('/user/<username>')
def show_user_profile(username):
    return jsonify({'username': username, 'message': f'Profile for {username}'})

@app.route('/post/<int:post_id>')
def show_post(post_id):
    return jsonify({'post_id': post_id, 'type': type(post_id).__name__})

@app.route('/category/<string:category_name>')
def show_category(category_name):
    return jsonify({'category': category_name})

@app.route('/file/<path:filename>')
def serve_file(filename):
    return jsonify({'filename': filename, 'path': f'/files/{filename}'})

@app.route('/product/<uuid:product_uuid>')
def show_product(product_uuid):
    return jsonify({'product_uuid': str(product_uuid)})

@app.route('/user/<username>/posts/<int:post_id>')
def user_post(username, post_id):
    return jsonify({
        'username': username,
        'post_id': post_id,
        'url': f'/{username}/posts/{post_id}'
    })

# Query parameters
@app.route('/search-demo')
def search_demo():
    query = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    tags = request.args.getlist('tags')  # Multiple values
    
    return jsonify({
        'query': query,
        'page': page,
        'per_page': per_page,
        'tags': tags,
        'all_args': dict(request.args)
    })

url_parameters_code = '''
from flask import request
from uuid import UUID

# Path parameters with types
@app.route('/user/<string:username>')
def user_profile(username):
    return f'User: {username}'

@app.route('/post/<int:post_id>')
def post_detail(post_id):
    # post_id is automatically converted to int
    return f'Post ID: {post_id} (type: {type(post_id)})'

@app.route('/price/<float:price>')
def price_info(price):
    return f'Price: ${price:.2f}'

@app.route('/file/<path:filepath>')
def serve_file(filepath):
    # path type allows slashes in the parameter
    return f'File: {filepath}'

@app.route('/uuid/<uuid:item_uuid>')
def item_by_uuid(item_uuid):
    return f'UUID: {item_uuid}'

# Multiple parameters
@app.route('/user/<username>/post/<int:post_id>')
def user_post(username, post_id):
    return f'User {username}, Post {post_id}'

# Optional parameters with defaults
@app.route('/blog/')
@app.route('/blog/<int:page>')
def blog(page=1):
    return f'Blog page: {page}'

# Query parameters
@app.route('/search')
def search():
    # Get single query parameter
    query = request.args.get('q', default='', type=str)
    page = request.args.get('page', default=1, type=int)
    
    # Get multiple values for same parameter
    tags = request.args.getlist('tags')
    
    # Get all arguments as dict
    all_args = request.args.to_dict()
    
    return {
        'query': query,
        'page': page,
        'tags': tags,
        'all_args': all_args
    }

# Form parameters (POST data)
@app.route('/submit', methods=['POST'])
def submit_form():
    # Get form data
    name = request.form.get('name')
    email = request.form.get('email')
    
    # Get multiple values (checkboxes)
    interests = request.form.getlist('interests')
    
    return {
        'name': name,
        'email': email,
        'interests': interests
    }

# JSON data from request body
@app.route('/api/data', methods=['POST'])
def handle_json():
    data = request.get_json()
    return {'received': data}

# Custom URL converters
from werkzeug.routing import BaseConverter

class ListConverter(BaseConverter):
    def to_python(self, value):
        return value.split(',')
    
    def to_url(self, values):
        return ','.join(BaseConverter.to_url(value) for value in values)

app.url_map.converters['list'] = ListConverter

@app.route('/tags/<list:tag_list>')
def show_tags(tag_list):
    return {'tags': tag_list}
'''
print("URL Parameters:")
print(url_parameters_code)

print("\n" + "="*80)
print("ADDITIONAL ADVANCED EXAMPLES")
print("="*80)

# Database Model Example
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat()
        }

# RESTful API using Flask-RESTful
class UserResource(Resource):
    def get(self, user_id=None):
        if user_id:
            user = User.query.get_or_404(user_id)
            return user.to_dict()
        else:
            users = User.query.all()
            return [user.to_dict() for user in users]
    
    def post(self):
        data = request.get_json()
        user = User(username=data['username'], email=data['email'])
        db.session.add(user)
        db.session.commit()
        return user.to_dict(), 201
    
    def put(self, user_id):
        user = User.query.get_or_404(user_id)
        data = request.get_json()
        user.username = data.get('username', user.username)
        user.email = data.get('email', user.email)
        db.session.commit()
        return user.to_dict()
    
    def delete(self, user_id):
        user = User.query.get_or_404(user_id)
        db.session.delete(user)
        db.session.commit()
        return {'message': 'User deleted'}, 200

# Register REST API endpoints
api.add_resource(UserResource, '/api/users', '/api/users/<int:user_id>')

# Complete Flask application setup
@app.before_first_request
def create_tables():
    db.create_all()

# CORS handling
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Rate limiting example (basic)
from collections import defaultdict
from time import time

request_counts = defaultdict(list)

def rate_limit(max_requests=100, window=3600):
    def decorator(f):
        def decorated_function(*args, **kwargs):
            now = time()
            client_ip = request.remote_addr
            
            # Clean old requests
            request_counts[client_ip] = [
                req_time for req_time in request_counts[client_ip] 
                if now - req_time < window
            ]
            
            if len(request_counts[client_ip]) >= max_requests:
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            request_counts[client_ip].append(now)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/api/limited')
@rate_limit(max_requests=10, window=60)  # 10 requests per minute
def limited_endpoint():
    return jsonify({'message': 'This endpoint is rate limited'})

# API versioning example
from flask import Blueprint

# Version 1 API
api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')

@api_v1.route('/users')
def v1_users():
    return jsonify({'version': '1.0', 'users': []})

# Version 2 API
api_v2 = Blueprint('api_v2', __name__, url_prefix='/api/v2')

@api_v2.route('/users')
def v2_users():
    return jsonify({'version': '2.0', 'users': [], 'new_features': ['pagination', 'filtering']})

app.register_blueprint(api_v1)
app.register_blueprint(api_v2)

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

# API documentation endpoint
@app.route('/api/docs')
def api_docs():
    return jsonify({
        'endpoints': {
            'GET /api/users': 'Get all users',
            'POST /api/users': 'Create a new user',
            'GET /api/users/<id>': 'Get user by ID',
            'PUT /api/users/<id>': 'Update user by ID',
            'DELETE /api/users/<id>': 'Delete user by ID',
            'GET /health': 'Health check',
            'GET /api/docs': 'API documentation'
        },
        'version': '1.0.0',
        'base_url': request.base_url.replace('/api/docs', '')
    })

print("\nAdvanced Flask features implemented!")
print("\n" + "="*80)
print("COMPLETE FLASK APPLICATION READY!")
print("="*80)

print("\nTo run this application:")
print("1. Install required packages: pip install flask flask-sqlalchemy flask-restful")
print("2. Copy this code to a file (e.g., app.py)")
print("3. Run with: python app.py")
print("4. Access endpoints like:")
print("   - http://localhost:5000/")
print("   - http://localhost:5000/api/users")
print("   - http://localhost:5000/health")
print("   - http://localhost:5000/api/docs")

# If running as main application
if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
    
    print("Starting Flask application...")
    print("Available endpoints:")
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods - {'HEAD', 'OPTIONS'})
        print(f"  {methods:10} {rule}")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)