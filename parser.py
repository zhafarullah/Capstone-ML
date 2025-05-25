# parser.py

import re

unit_conversion = {
    'mg': 0.001, 'milligram': 0.001, 'milligrams': 0.001,
    'g': 1, 'gram': 1, 'grams': 1,
    'kg': 1000, 'kilo': 1000, 'kilogram': 1000, 'kilograms': 1000,
    'oz': 28.3495, 'ounce': 28.3495, 'ounces': 28.3495,
    'lb': 453.592, 'lbs': 453.592, 'pound': 453.592, 'pounds': 453.592,
    'ml': 1, 'milliliter': 1, 'milliliters': 1, 'millilitre': 1, 'millilitres': 1,
    'l': 1000, 'liter': 1000, 'litre': 1000, 'liters': 1000, 'litres': 1000,
    'dl': 100, 'deciliter': 100, 'deciliters': 100, 'decilitre': 100, 'decilitres': 100,
    'tsp': 4.92892, 'teaspoon': 4.92892, 'teaspoons': 4.92892,
    'tbsp': 14.7868, 'tablespoon': 14.7868, 'tablespoons': 14.7868,
    'cup': 240, 'cups': 240, 'c': 240,
    'pt': 473.176, 'pint': 473.176, 'pints': 473.176,
    'qt': 946.353, 'quart': 946.353, 'quarts': 946.353,
    'gal': 3785.41, 'gallon': 3785.41, 'gallons': 3785.41,
}

# Buat regex pattern unit dinamis berdasarkan keys unit_conversion
unit_pattern = '|'.join(unit_conversion.keys())

def parse_ingredient(input_text):
    pattern = rf'(\d+(?:\.\d+)?)\s*({unit_pattern})\s+(?:of\s+)?([a-zA-Z\s]+?)(?:,|and|$)'
    matches = re.findall(pattern, input_text.lower())

    result = []
    for qty, unit, ingredient in matches:
        qty = float(qty) * unit_conversion.get(unit, 1)
        result.append({
            "ingredient": ingredient.strip(),
            "quantity": qty,  # jangan dibulatkan dulu biar presisi
            "unit": "gram"  # semua dikonversi ke gram
        })
    return result
