#read restaurants
import pandas as pd
MENU_PATH = "data/restaurant-menus.csv"

df = pd.read_csv(MENU_PATH, nrows =20)
df = df.fillna("")

menu_map = {}

for _, row in df.iterrows():
    restaurant_id = row["restaurant_id"]
    menu_text = f"{row['name']} | {row['category']} | {row['description']} | {row['price']}"

    if restaurant_id not in menu_map:
        menu_map[restaurant_id] = []

    if menu_text not in menu_map[restaurant_id] and len(menu_map[restaurant_id]) < 10:
        menu_map[restaurant_id].append(menu_text)

print("Restaurant 1 menu items:")
for item in menu_map.get(1, []):
    print("-", item)
