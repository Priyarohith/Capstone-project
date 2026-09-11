import sqlite3
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/catalogue/"
FIXED_RATE_INR_PER_GBP = 105.50
MAX_PAGES = 3

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "zepto_books.db"
QUERY_LOG_PATH = ROOT / "query_outputs.txt"


def fetch_catalogue_page(page_num: int = 1):
    url = f"{BASE_URL}page-{page_num}.html"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.text


def parse_star_rating(star_rating_tag):
    text = star_rating_tag.get("class", [])
    rating_text = [value for value in text if value.lower() in {"one", "two", "three", "four", "five"}]
    if not rating_text:
        return None
    mapping = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    return mapping[rating_text[0].lower()]


def parse_book_row(card, category_name):
    title_tag = card.select_one("h3 a")
    title = title_tag.get_text(strip=True) if title_tag else None

    price_raw = card.select_one(".price_color")
    price_text = price_raw.get_text(strip=True) if price_raw else None
    price_gbp = None
    if price_text:
        cleaned = price_text.replace("Â£", "").replace("£", "").replace("\xa3", "").replace(",", "").strip()
        try:
            price_gbp = float(cleaned)
        except ValueError:
            price_gbp = None

    rating_tag = card.select_one(".star-rating")
    rating = parse_star_rating(rating_tag) if rating_tag else None

    availability_tag = card.select_one(".availability")
    availability_text = availability_tag.get_text(" ", strip=True) if availability_tag else None
    in_stock = None
    if availability_text:
        in_stock = "in stock" in availability_text.lower()

    return {
        "title": title,
        "price_gbp": price_gbp,
        "rating": rating,
        "in_stock": in_stock,
        "category": category_name,
    }


def scrape_books():
    all_rows = []
    for page_num in range(1, MAX_PAGES + 1):
        html = fetch_catalogue_page(page_num)
        soup = BeautifulSoup(html, "html.parser")
        product_cards = soup.select("article.product_pod")
        if not product_cards:
            break
        for card in product_cards:
            row = parse_book_row(card, "all_products")
            if row["title"]:
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["title", "price_gbp", "rating", "in_stock", "category"]).reset_index(drop=True)
    df["price_gbp"] = df["price_gbp"].astype(float)
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    df["price_inr"] = df["price_gbp"] * FIXED_RATE_INR_PER_GBP
    return df


def create_schema(conn):
    conn.execute("DROP TABLE IF EXISTS books")
    conn.execute("DROP TABLE IF EXISTS categories")
    conn.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER REFERENCES categories(category_id)
        )
        """
    )


def load_into_db(df):
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)

    category_df = pd.DataFrame({"category_name": sorted(df["category"].unique())})
    category_df.to_sql("categories", conn, if_exists="append", index=False)
    category_map = pd.read_sql(
        "SELECT category_name, rowid AS category_id FROM categories",
        conn,
    )
    category_lookup = dict(zip(category_map["category_name"], category_map["category_id"]))

    book_rows = df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]].copy()
    book_rows["category_id"] = book_rows["category"].map(category_lookup)
    book_rows = book_rows[["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]]
    book_rows.columns = ["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]
    book_rows.to_sql("books", conn, if_exists="append", index=False)
    conn.close()


def run_queries(conn):
    queries = [
        (
            "SELECT * FROM books WHERE in_stock = 1 ORDER BY rating DESC LIMIT 10;",
            "SELECT/WHERE + ORDER BY + LIMIT",
        ),
        (
            "SELECT title, price_gbp, price_inr FROM books ORDER BY price_gbp DESC LIMIT 10;",
            "ORDER BY + LIMIT",
        ),
        (
            "SELECT DISTINCT category_id FROM books;",
            "DISTINCT",
        ),
        (
            "SELECT title, price_inr FROM books WHERE price_inr BETWEEN 400 AND 800 ORDER BY price_inr DESC;",
            "BETWEEN",
        ),
        (
            "SELECT c.category_name, COUNT(b.book_id) AS book_count FROM books b JOIN categories c ON b.category_id = c.category_id GROUP BY c.category_name ORDER BY book_count DESC;",
            "JOIN + GROUP BY",
        ),
        (
            "SELECT title, rating FROM books WHERE rating IN (4, 5) ORDER BY rating DESC LIMIT 10;",
            "IN",
        ),
    ]

    results = []
    for statement, label in queries:
        result = pd.read_sql(statement, conn)
        print(f"\n--- {label} ---\n{statement}\n")
        print(result.head(10).to_string(index=False))
        results.append((label, statement, result))

    return results


def demo_pd_read_sql_and_merge(conn):
    join_sql = """
        SELECT c.category_name, b.title, b.rating, b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY c.category_name, b.rating DESC
    """
    sql_result = pd.read_sql(join_sql, conn)

    books_df = pd.read_sql("SELECT * FROM books", conn)
    categories_df = pd.read_sql("SELECT * FROM categories", conn)
    merged_df = books_df.merge(categories_df, left_on="category_id", right_on="category_id", how="left")
    merged_df = merged_df[["category_name", "title", "rating", "price_inr"]].sort_values(["category_name", "rating"], ascending=[True, False])

    print("\n--- pd.read_sql join output ---")
    print(sql_result.head(10).to_string(index=False))
    print("\n--- pd.merge equivalent output ---")
    print(merged_df.head(10).to_string(index=False))
    print("\nEquivalent output check:", sql_result.reset_index(drop=True).equals(merged_df.reset_index(drop=True)))


def main():
    df = scrape_books()
    print(f"Scraped {len(df)} books across categories: {sorted(df['category'].unique())}")

    if len(df) < 60:
        raise ValueError("Expected at least 60 rows after scraping and cleaning.")

    load_into_db(df)

    conn = sqlite3.connect(DB_PATH)
    query_results = run_queries(conn)
    demo_pd_read_sql_and_merge(conn)
    conn.close()

    with QUERY_LOG_PATH.open("w", encoding="utf-8") as fh:
        for label, statement, result in query_results:
            fh.write(f"--- {label} ---\n{statement}\n\n")
            fh.write(result.head(20).to_string(index=False))
            fh.write("\n\n")

    print(f"\nDatabase saved to: {DB_PATH}")
    print(f"Query logs saved to: {QUERY_LOG_PATH}")


if __name__ == "__main__":
    main()
