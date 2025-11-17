#!/usr/bin/env python3
"""
Generate synthetic SQL Micro-Brain training examples.
Creates original examples for rental/booking, e-commerce, and general SQL themes.
"""

import argparse
import json
import random
import sys
from pathlib import Path

def generate_rental_example(i):
    themes = [
        {"type": "wedding_venue", "name": "wedding venues", "task": "Find wedding venues in Jakarta available on 2024-12-15 for 100-200 people.", "schema": 'CREATE TABLE venues (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), name text, city text, capacity integer, price_per_hour numeric(10,2)); CREATE TABLE bookings (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), venue_id uuid REFERENCES venues(id), customer_id uuid, booked_date date, start_time time, end_time time);', "actions": ["select", "add_indexes"], "indexes": [{"sql": "CREATE INDEX idx_venues_city_capacity ON venues (city, capacity);"}, {"sql": "CREATE INDEX idx_bookings_venue_date ON bookings (venue_id, booked_date);"}], "queries": [{"description": "Available venues", "sql": "SELECT v.id, v.name FROM venues v WHERE v.city = 'Jakarta' AND v.capacity BETWEEN 100 AND 200 AND NOT EXISTS (SELECT 1 FROM bookings b WHERE b.venue_id = v.id AND b.booked_date = '2024-12-15');"}], "explanations": ["Checked availability using NOT EXISTS."]},
        {"type": "tuxedo", "name": "tuxedo rentals", "task": "Check if a specific tuxedo is available for rental from 2024-11-01 to 2024-11-05.", "schema": 'CREATE TABLE tuxedos (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), size text, color text, rental_price numeric(7,2)); CREATE TABLE rentals (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), tuxedo_id uuid REFERENCES tuxedos(id), renter_id uuid, start_date date, end_date date);', "actions": ["select"], "indexes": [{"sql": "CREATE INDEX idx_rentals_tuxedo_dates ON rentals (tuxedo_id, start_date, end_date);"}], "queries": [{"description": "Tuxedo availability", "sql": "SELECT CASE WHEN EXISTS (SELECT 1 FROM rentals r WHERE r.tuxedo_id = $1 AND (r.start_date, r.end_date) OVERLAPS ('2024-11-01'::date, '2024-11-05'::date)) THEN false ELSE true END AS available;"}], "explanations": ["Used OVERLAPS for date range checking."]},
        {"type": "salon", "name": "salon appointments", "task": "Book a haircut appointment tomorrow if slot is free.", "schema": 'CREATE TABLE services (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), name text, price numeric(7,2)); CREATE TABLE appointments (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), customer_id uuid, service_id uuid REFERENCES services(id), appointment_time timestamptz); CREATE TABLE slots (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), start_time timestamptz, available boolean DEFAULT true);', "actions": ["insert", "select"], "indexes": [{"sql": "CREATE INDEX idx_appointments_time ON appointments (appointment_time);"}], "queries": [{"description": "Conditional insert", "sql": "INSERT INTO appointments (customer_id, service_id, appointment_time) SELECT $1, $2, $3 WHERE NOT EXISTS (SELECT 1 FROM appointments a WHERE a.appointment_time = $3);"}], "explanations": ["Inserted only if no conflicting appointment."]},
        {"type": "gym", "name": "gym classes", "task": "List gym classes with current enrollment count next week.", "schema": 'CREATE TABLE classes (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), instructor_id uuid, name text, schedule timestamptz, max_capacity integer); CREATE TABLE enrollments (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), class_id uuid REFERENCES classes(id), student_id uuid);', "actions": ["select"], "indexes": [{"sql": "CREATE INDEX idx_enrollments_class ON enrollments (class_id);"}], "queries": [{"description": "Classes with enrollment counts", "sql": "SELECT c.name, c.schedule, count(e.id) as enrolled FROM classes c LEFT JOIN enrollments e ON c.id = e.class_id WHERE c.schedule >= current_timestamp + interval '1 day' AND c.schedule < current_timestamp + interval '1 week' GROUP BY c.id, c.name, c.schedule HAVING count(e.id) < c.max_capacity;"}], "explanations": ["Used LEFT JOIN and HAVING for filtered counts."]},
        {"type": "villas", "name": "villa rentals", "task": "Find villas available for booking from 2024-12-20 to 2024-12-25.", "schema": 'CREATE TABLE villas (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), name text, location text, max_guests integer); CREATE TABLE rentals (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), villa_id uuid REFERENCES villas(id), start_date date, end_date date);', "actions": ["select"], "indexes": [{"sql": "CREATE INDEX idx_rentals_villa_dates ON rentals (villa_id, start_date, end_date);"}], "queries": [{"description": "Available villas", "sql": "SELECT v.id, v.name FROM villas v WHERE v.max_guests >= 4 AND NOT EXISTS (SELECT 1 FROM rentals r WHERE r.villa_id = v.id AND (r.start_date, r.end_date) OVERLAPS ('2024-12-20'::date, '2024-12-25'::date));"}], "explanations": ["Checked for overlaps in rental dates."]},
        {"type": "classroom", "name": "classroom bookings", "task": "Book a classroom for a workshop on 2024-10-15 from 2 PM to 4 PM.", "schema": 'CREATE TABLE classrooms (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), name text, capacity integer); CREATE TABLE bookings (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), classroom_id uuid REFERENCES classrooms(id), event_name text, start_time timestamptz, end_time timestamptz);', "actions": ["insert"], "indexes": [{"sql": "CREATE INDEX idx_bookings_classroom_time ON bookings (classroom_id, start_time, end_time);"}], "queries": [{"description": "Insert booking", "sql": "INSERT INTO bookings (classroom_id, event_name, start_time, end_time) VALUES ($1, 'Workshop', '2024-10-15 14:00:00'::timestamptz, '2024-10-15 16:00:00'::timestamptz);"}], "explanations": ["Basic insert for booking."]},
        {"type": "coworking", "name": "coworking desks", "task": "Find free coworking desks in Zone A tomorrow.", "schema": 'CREATE TABLE desks (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), zone text, hourly_rate numeric(5,2)); CREATE TABLE reservations (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), desk_id uuid REFERENCES desks(id), user_id uuid, reserved_date date, start_time time, end_time time);', "actions": ["select"], "indexes": [{"sql": "CREATE INDEX idx_desks_zone ON desks (zone);"}, {"sql": "CREATE INDEX idx_reservations_desk_date ON reservations (desk_id, reserved_date);"}], "queries": [{"description": "Free desks tomorrow", "sql": "SELECT d.id, d.hourly_rate FROM desks d WHERE d.zone = 'A' AND NOT EXISTS (SELECT 1 FROM reservations r WHERE r.desk_id = d.id AND r.reserved_date = current_date + interval '1 day');"}], "explanations": ["Assumed full day booking for simplicity."]}
    ]
    theme = random.choice(themes)

    return {
        "id": f"{theme['type']}_{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))}_{i:04d}",
        "schema": theme['schema'],
        "task": theme['task'],
        "output": {
            "actions": theme['actions'],
            "migrations": [],
            "rls_policies": [],
            "indexes": theme['indexes'],
            "queries": theme['queries'],
            "error_explanations": [],
            "explanations": theme['explanations'],
            "safe_to_execute": True
        }
    }

def generate_ecommerce_example(i):
    themes = [
        {"name": "bestselling products", "task": "List top 10 best-selling products over the last 90 days.", "indexes": [{"sql": "CREATE INDEX idx_order_items_product ON order_items (product_id);"}, {"sql": "CREATE INDEX idx_orders_date ON orders (order_date);"}], "queries": [{"description": "Top products by sales", "sql": "SELECT p.name, sum(oi.quantity) as total_sold FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.order_date >= current_date - interval '90 days' GROUP BY p.id, p.name ORDER BY total_sold DESC LIMIT 10;"}]},
        {"name": "abandoned carts", "task": "Find customers with abandoned carts (not updated in the last 7 days).", "indexes": [{"sql": "CREATE INDEX idx_carts_updated ON carts (updated_at);"}], "queries": [{"description": "Abandoned carts with details", "sql": "SELECT c.customer_id, array_agg(ci.product_id) as product_ids FROM carts c JOIN cart_items ci ON c.id = ci.cart_id WHERE c.updated_at < current_timestamp - interval '7 days' GROUP BY c.customer_id;"}]},
        {"name": "product search", "task": "Find products by name and category matching a query.", "indexes": [{"sql": "CREATE INDEX idx_products_name_category ON products (name, category);"}], "queries": [{"description": "Search products", "sql": "SELECT p.id, p.name, p.price FROM products p WHERE p.name ILIKE '%keyword%' AND p.category = 'electronics' ORDER BY p.price ASC LIMIT 50;"}]}
    ]
    theme = random.choice(themes)
    schema = f'CREATE TABLE products (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), name text, category text, price numeric(8,2)); CREATE TABLE orders (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), customer_id uuid, order_date date); CREATE TABLE order_items (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), order_id uuid REFERENCES orders(id), product_id uuid REFERENCES products(id), quantity integer); CREATE TABLE carts (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), customer_id uuid, updated_at timestamptz); CREATE TABLE cart_items (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), cart_id uuid REFERENCES carts(id), product_id uuid, quantity integer);'

    return {
        "id": f"ecommerce_{theme['name'].replace(' ', '_')}_{i:04d}",
        "schema": schema,
        "task": theme["task"],
        "output": {
            "actions": ["select", "add_indexes"] if "add_indexes" in theme.get("actions", []) else ["select"],
            "migrations": [],
            "rls_policies": [],
            "indexes": theme["indexes"],
            "queries": theme["queries"],
            "error_explanations": [],
            "explanations": ["Used text search with ILIKE and filtering by category."],
            "safe_to_execute": True
        }
    }

def generate_general_sql_example(i):
    features = [
        {"name": "window_functions", "task": "Find the last booking per customer using window functions.", "queries": [{"description": "Last booking per customer", "sql": "SELECT id, customer_id, booking_date FROM (SELECT id, customer_id, booking_date, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY booking_date DESC) as rn FROM bookings) sub WHERE rn = 1;"}], "explanations": ["Used ROW_NUMBER window function to rank bookings per customer."]},
        {"name": "joins_and_aggregates", "task": "Calculate the total order value for each customer.", "queries": [{"description": "Customer order totals", "sql": "SELECT c.name, sum(o.total_amount) as total FROM customers c LEFT JOIN orders o ON c.id = o.customer_id WHERE o.status = 'completed' GROUP BY c.id, c.name ORDER BY total DESC;"}], "explanations": ["Used LEFT JOIN and aggregate sum for totals."]},
        {"name": "partial_indexes", "task": "Add a partial index to speed up queries on active products.", "queries": [], "explanations": ["Added partial index for active products to improve query performance."]},
    ]
    feature = random.choice(features)
    schema = f'CREATE TABLE customers (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), name text, active boolean DEFAULT true); CREATE TABLE bookings (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), customer_id uuid REFERENCES customers(id), booking_date date, amount numeric(8,2)); CREATE TABLE orders (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), customer_id uuid, total_amount numeric(8,2), status text); CREATE TABLE products (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), name text, category text, active boolean DEFAULT true);'
    return {
        "id": f"general_sql_{feature['name']}_{i:04d}",
        "schema": schema,
        "task": feature["task"],
        "output": {
            "actions": ["select", "add_indexes"] if feature["queries"] else ["add_indexes"],
            "migrations": [],
            "rls_policies": [],
            "indexes": [{"sql": "CREATE INDEX CONCURRENTLY idx_active_products_name ON products (name) WHERE active = true;"}] if feature["name"] == "partial_indexes" else [{"sql": "CREATE INDEX idx_bookings_customer ON bookings (customer_id);"}],
            "queries": feature["queries"],
            "error_explanations": [],
            "explanations": feature["explanations"],
            "safe_to_execute": True
        }
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic SQL Micro-Brain training examples")
    parser.add_argument("--count", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="data/raw/synthetic_examples.jsonl", help="Output file")

    args = parser.parse_args(argv)

    print(f"Generating {args.count} synthetic examples...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for i in range(args.count):
            # Mix themes: 40% rental, 40% ecommerce, 20% general
            rand = random.randint(1, 100)
            if rand <= 40:
                example = generate_rental_example(i)
            elif rand <= 80:
                example = generate_ecommerce_example(i)
            else:
                example = generate_general_sql_example(i)

            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Generated {args.count} synthetic examples in {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
