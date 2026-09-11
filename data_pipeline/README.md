# Data Pipeline Module

This module handles the full data-engineering workflow for Zepto's catalog intelligence pipeline: scrape, clean, convert, normalize, store, and query.

## Data source

The scraper uses the public books.toscrape.com catalog, which is intentionally built for scraping practice. The pipeline collects product records from multiple categories and stores them in a normalized SQLite database.

## Fixed conversion rule

The required project definition is:

- 1 GBP = 105.50 INR

This is the fixed-rate baseline used for all currency conversion in this repository. No API lookup is required.

## Cleaning choices

- Price strings are stripped of the GBP symbol and converted to float values.
- Star ratings are mapped from text values like "Three" to integer values from 1 to 5.
- Availability text is converted to a boolean using the in-stock phrase.
- Any row with an unparseable critical field is dropped rather than imputed, because a malformed catalog record is better omitted than silently invented in a product-data workflow.

## Database schema

The database uses a two-table relational design:

- categories(category_id PRIMARY KEY, category_name UNIQUE)
- books(book_id PRIMARY KEY, title, price_gbp, price_inr, rating, in_stock, category_id REFERENCES categories)

This ensures normalized data storage instead of denormalized repeated category names.

## Run instructions

```bash
cd data_pipeline
python scrape_books.py
```

This script:

1. Scrapes at least 60 books across multiple categories.
2. Cleans and converts the fields.
3. Writes the relational SQLite database.
4. Executes at least 5 SQL queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, and JOIN.
5. Reads query outputs back to pandas DataFrames and compares a SQL join with a pandas merge.

## Output notes

The generated SQLite database is saved as [data_pipeline/zepto_books.db](zepto_books.db). Query output logs are stored in [data_pipeline/query_outputs.txt](query_outputs.txt).
