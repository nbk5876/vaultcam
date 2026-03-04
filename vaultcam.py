"""
VaultCam — AI-Powered Multi-Category Inventory App
Flask application for photographing items and building searchable inventory using GPT-4o Vision.
"""

from flask import (Flask, render_template, request, jsonify, session,
                   redirect, url_for, flash)
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
import json
import os
import base64
import bcrypt
from email_validator import validate_email, EmailNotValidError
import bleach
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))

# Database configuration — Render provides postgres:// but SQLAlchemy requires postgresql://
db_url = os.getenv("DATABASE_URL", "")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
db = SQLAlchemy(app)


# --------------------------------------------------
# Database Models
# --------------------------------------------------
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(200), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class Category(db.Model):
    __tablename__ = 'categories'
    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(50), unique=True, nullable=False)
    display_name = db.Column(db.String(100), nullable=False)
    icon = db.Column(db.String(10))
    ai_prompt = db.Column(db.Text, nullable=False)
    is_active = db.Column(db.Boolean, default=True)


class Item(db.Model):
    __tablename__ = 'items'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.id'), nullable=False)
    name = db.Column(db.String(200))
    brand = db.Column(db.String(100))
    properties = db.Column(db.JSON)
    status = db.Column(db.String(30))
    image_data = db.Column(db.Text)
    ai_confidence = db.Column(db.String(10))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, onupdate=lambda: datetime.now(timezone.utc))
    user = db.relationship('User', backref='items')
    category = db.relationship('Category', backref='items')


# --------------------------------------------------
# Image Compression
# --------------------------------------------------
def compress_image(image_file, max_size=800, quality=75):
    """Resize and compress image to JPEG. Returns bytes."""
    img = Image.open(image_file).convert('RGB')
    img.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality, optimize=True)
    return buffer.getvalue()


# --------------------------------------------------
# Seed Data
# --------------------------------------------------
def seed_categories():
    nail_polish_prompt = '''Analyze this nail polish bottle.
Return ONLY a JSON object, no other text:
{"brand": "brand name or null",
 "color_name": "color name from label or null",
 "finish": "one of: creme/glitter/shimmer/matte/metallic/gel or null",
 "color_hex": "best estimate hex like #FF5733 or null",
 "confidence": "high/medium/low"}'''

    spice_prompt = '''Analyze this spice jar/package.
Return ONLY a JSON object, no other text:
{"name": "spice name or null",
 "brand": "brand name or null",
 "origin": "country or region of origin or null",
 "heat_level": "one of: none/mild/medium/hot/extra_hot or null",
 "format": "one of: whole/ground/flake/blend or null",
 "expiry_date": "MM/YYYY format or null",
 "confidence": "high/medium/low"}'''

    album_prompt = '''Analyze this vinyl record album cover or label.
Return ONLY a JSON object, no other text:
{"artist": "artist or band name or null",
 "title": "album title or null",
 "label": "record label name or null",
 "year": "release year as 4-digit string or null",
 "genre": "one of: rock/pop/jazz/blues/classical/country/soul/funk/folk/electronic/other or null",
 "condition": "one of: mint/near_mint/very_good_plus/very_good/good/fair/poor or null",
 "pressing": "one of: original/reissue/repress/unknown or null",
 "color_variant": "black/colored/picture_disc or null",
 "estimated_value": estimated collector market value in USD as a number or null,
 "confidence": "high/medium/low"}'''

    if not Category.query.filter_by(slug='vinyl_album').first():
        db.session.add(Category(
            slug='vinyl_album', display_name='Vinyl Albums',
            icon='\U0001f3b5', ai_prompt=album_prompt))

    if not Category.query.filter_by(slug='nail_polish').first():
        db.session.add(Category(
            slug='nail_polish', display_name='Nail Polish',
            icon='\U0001f485', ai_prompt=nail_polish_prompt))
    if not Category.query.filter_by(slug='spice').first():
        db.session.add(Category(
            slug='spice', display_name='Spices',
            icon='\U0001f336\ufe0f', ai_prompt=spice_prompt))
    db.session.commit()


# TEMPORARY: Guest account for dev/testing — remove before production launch
GUEST_EMAIL = 'guest@vaultcam.app'
GUEST_NAME = 'Guest'


def ensure_guest_user():
    """Create the shared guest account if it doesn't exist."""
    guest = User.query.filter_by(email=GUEST_EMAIL).first()
    if not guest:
        guest = User(
            email=GUEST_EMAIL,
            name=GUEST_NAME,
            password_hash='no-login'  # not a real hash — guest bypasses password auth
        )
        db.session.add(guest)
        db.session.commit()
    return guest


# --------------------------------------------------
# Auth Helper
# --------------------------------------------------
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('landing'))
        return f(*args, **kwargs)
    return decorated


def is_guest():
    """Return True if the current session belongs to the guest account."""
    return session.get('is_guest', False)


@app.context_processor
def inject_guest_flag():
    """Make guest_mode available in all templates."""
    return {'guest_mode': is_guest()}


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route('/')
def landing():
    if session.get('user_id'):
        return redirect(url_for('dashboard'))
    return render_template('landing.html')


# TEMPORARY: Guest access route for dev/testing
@app.route('/guest')
def guest_login():
    guest = User.query.filter_by(email=GUEST_EMAIL).first()
    if not guest:
        guest = ensure_guest_user()
    session['user_id'] = guest.id
    session['user_name'] = guest.name
    session['user_email'] = guest.email
    session['is_guest'] = True
    return redirect(url_for('dashboard'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = bleach.clean(request.form.get('name', '').strip())
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not name or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('landing.html', show_signup=True)

        try:
            valid = validate_email(email, check_deliverability=False)
            email = valid.normalized
        except EmailNotValidError:
            flash('Please enter a valid email address.', 'error')
            return render_template('landing.html', show_signup=True)

        if len(password) < 8:
            flash('Password must be at least 8 characters.', 'error')
            return render_template('landing.html', show_signup=True)

        if User.query.filter_by(email=email).first():
            flash('An account with that email already exists.', 'error')
            return render_template('landing.html', show_signup=True)

        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        user = User(email=email, name=name, password_hash=password_hash)
        db.session.add(user)
        db.session.commit()

        session['user_id'] = user.id
        session['user_name'] = user.name
        session['user_email'] = user.email
        return redirect(url_for('dashboard'))

    return render_template('landing.html', show_signup=True)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['user_email'] = user.email
            return redirect(url_for('dashboard'))

        flash('Invalid email or password.', 'error')
        return render_template('landing.html', show_login=True)

    return render_template('landing.html', show_login=True)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))


@app.route('/dashboard')
@login_required
def dashboard():
    category_slug = request.args.get('category')
    categories = Category.query.filter_by(is_active=True).all()

    query = Item.query.filter_by(user_id=session['user_id'])
    if category_slug:
        cat = Category.query.filter_by(slug=category_slug).first()
        if cat:
            query = query.filter_by(category_id=cat.id)

    items = query.order_by(
        db.cast(
            db.func.nullif(db.cast(Item.properties['estimated_value'], db.Text), 'null'),
            db.Numeric
        ).desc().nullslast(),
        Item.brand.asc(),
        Item.name.asc()
    ).all()
    return render_template('dashboard.html', items=items, categories=categories,
                           active_category=category_slug)


@app.route('/add')
@login_required
def add():
    if is_guest():
        flash('Guest accounts are read-only. Sign up to add items!', 'error')
        return redirect(url_for('dashboard'))
    categories = Category.query.filter_by(is_active=True).all()
    return render_template('add.html', categories=categories)


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if is_guest():
        return jsonify({'error': 'Guest accounts are read-only'}), 403
    category_slug = request.form.get('category_slug')
    image_file = request.files.get('image')

    if not category_slug or not image_file:
        return jsonify({'error': 'Category and image are required'}), 400

    category = Category.query.filter_by(slug=category_slug).first_or_404()

    compressed_bytes = compress_image(image_file)
    image_data = base64.b64encode(compressed_bytes).decode('utf-8')
    mime_type = 'image/jpeg'  # always JPEG after compression

    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{'role': 'user', 'content': [
            {'type': 'image_url',
             'image_url': {'url': f'data:{mime_type};base64,{image_data}'}},
            {'type': 'text', 'text': category.ai_prompt}
        ]}],
        max_tokens=300
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace('```json', '').replace('```', '').strip()
    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        return jsonify({'error': 'AI response could not be parsed', 'raw': raw}), 500

    return jsonify({
        'extracted': extracted,
        'image_data': f'data:{mime_type};base64,{image_data}',
        'category_slug': category_slug,
        'category_id': category.id
    })


@app.route('/items', methods=['POST'])
@login_required
def save_item():
    if is_guest():
        return jsonify({'error': 'Guest accounts are read-only'}), 403
    category_id = request.form.get('category_id', type=int)
    category = Category.query.get_or_404(category_id)

    name = bleach.clean(request.form.get('name', '').strip())
    brand = bleach.clean(request.form.get('brand', '').strip())
    status = request.form.get('status', 'owned')
    notes = bleach.clean(request.form.get('notes', '').strip())
    image_data = request.form.get('image_data', '')
    ai_confidence = request.form.get('ai_confidence', '')

    # Build properties JSON from category-specific fields
    properties = {}
    if category.slug == 'nail_polish':
        properties['finish'] = request.form.get('finish', '')
        properties['color_hex'] = request.form.get('color_hex', '')
    elif category.slug == 'spice':
        properties['heat_level'] = request.form.get('heat_level', '')
        properties['origin'] = request.form.get('origin', '')
        properties['format'] = request.form.get('format', '')
        properties['expiry_date'] = request.form.get('expiry_date', '')
    elif category.slug == 'vinyl_album':
        properties['label'] = request.form.get('label', '')
        properties['year'] = request.form.get('year', '')
        properties['genre'] = request.form.get('genre', '')
        properties['condition'] = request.form.get('condition', '')
        properties['pressing'] = request.form.get('pressing', '')
        properties['color_variant'] = request.form.get('color_variant', '')
        ev = request.form.get('estimated_value', '')
        properties['estimated_value'] = float(ev) if ev else None

    item = Item(
        user_id=session['user_id'],
        category_id=category_id,
        name=name,
        brand=brand,
        properties=properties,
        status=status,
        image_data=image_data,
        ai_confidence=ai_confidence,
        notes=notes
    )
    db.session.add(item)
    db.session.commit()
    flash('Item saved!', 'success')
    return redirect(url_for('dashboard'))


@app.route('/items/<int:item_id>')
@login_required
def item_detail(item_id):
    item = Item.query.get_or_404(item_id)
    if item.user_id != session['user_id']:
        return redirect(url_for('dashboard'))
    return render_template('item_detail.html', item=item)


@app.route('/items/<int:item_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_item(item_id):
    if is_guest():
        flash('Guest accounts are read-only. Sign up to edit items!', 'error')
        return redirect(url_for('item_detail', item_id=item_id))
    item = Item.query.get_or_404(item_id)
    if item.user_id != session['user_id']:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        item.name = bleach.clean(request.form.get('name', '').strip())
        item.brand = bleach.clean(request.form.get('brand', '').strip())
        item.status = request.form.get('status', item.status)
        item.notes = bleach.clean(request.form.get('notes', '').strip())

        properties = item.properties or {}
        if item.category.slug == 'nail_polish':
            properties['finish'] = request.form.get('finish', '')
            properties['color_hex'] = request.form.get('color_hex', '')
        elif item.category.slug == 'spice':
            properties['heat_level'] = request.form.get('heat_level', '')
            properties['origin'] = request.form.get('origin', '')
            properties['format'] = request.form.get('format', '')
            properties['expiry_date'] = request.form.get('expiry_date', '')
        elif item.category.slug == 'vinyl_album':
            properties['label'] = request.form.get('label', '')
            properties['year'] = request.form.get('year', '')
            properties['genre'] = request.form.get('genre', '')
            properties['condition'] = request.form.get('condition', '')
            properties['pressing'] = request.form.get('pressing', '')
            properties['color_variant'] = request.form.get('color_variant', '')
            ev = request.form.get('estimated_value', '')
            properties['estimated_value'] = float(ev) if ev else None
        item.properties = properties

        db.session.commit()
        flash('Item updated!', 'success')
        return redirect(url_for('item_detail', item_id=item.id))

    categories = Category.query.filter_by(is_active=True).all()
    return render_template('item_detail.html', item=item, editing=True, categories=categories)


@app.route('/items/<int:item_id>/delete', methods=['POST'])
@login_required
def delete_item(item_id):
    if is_guest():
        return jsonify({'error': 'Guest accounts are read-only'}), 403
    item = Item.query.get_or_404(item_id)
    if item.user_id != session['user_id']:
        return redirect(url_for('dashboard'))
    db.session.delete(item)
    db.session.commit()
    flash('Item deleted.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/categories/<slug>')
@login_required
def category_view(slug):
    category = Category.query.filter_by(slug=slug).first_or_404()
    items = Item.query.filter_by(
        user_id=session['user_id'], category_id=category.id
    ).order_by(Item.created_at.desc()).all()
    categories = Category.query.filter_by(is_active=True).all()
    return render_template('dashboard.html', items=items, categories=categories,
                           active_category=slug)


@app.route('/search')
@login_required
def search():
    q = request.args.get('q', '').strip()
    items = []
    if q:
        search_term = f'%{q}%'
        items = Item.query.filter(
            Item.user_id == session['user_id'],
            db.or_(
                Item.name.ilike(search_term),
                Item.brand.ilike(search_term)
            )
        ).order_by(Item.created_at.desc()).all()
    categories = Category.query.filter_by(is_active=True).all()
    return render_template('search.html', items=items, query=q, categories=categories)


# --------------------------------------------------
# App Initialization
# --------------------------------------------------
with app.app_context():
    db.create_all()
    seed_categories()
    ensure_guest_user()  # TEMPORARY: seed guest account for dev/testing


if __name__ == '__main__':
    app.run(debug=True)
