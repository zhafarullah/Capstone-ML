from carbon_calculator import calculate_carbon_for_ingredients

def main():
    print("Masukkan bahan makanan dengan format seperti: '100 gram chicken, 50 gram rice'")
    user_input = input("Input bahan makanan: ")

    df_result = calculate_carbon_for_ingredients(user_input)

    total_cf = df_result["carbon_score"].sum()

    print("\nHasil perhitungan karbon:")
    print(df_result[["name", "quantity", "unit", "carbon_item", "carbon_score", "estimated_items", "matched_items"]])

    print(f"Total jejak karbon: {round(total_cf, 3)} kg CO2")


if __name__ == "__main__":
    main()
