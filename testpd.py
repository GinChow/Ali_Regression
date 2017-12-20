import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as dates
df = pd.read_csv("sales_train_v2.csv")

items = pd.read_csv("items.csv")
item_cate = pd.read_csv("item_categories.csv")

df.date = pd.to_datetime(df.date, format='%d.%m.%Y')


new_df = pd.merge(df, items, on='item_id')
new_df = new_df.loc[(new_df['date'].dt.year == 2014) & (new_df['date'].dt.month == 12) &(new_df['shop_id'] == 25)]


# print(df.l['date'].year)
# l = df.loc[df['date'].year == 2014]

# df['day'], df['month'], df['year'] = df['date']..split('-')
# print(df.head(1))

# filtered = df.loc[(df['date'].dt.year == 2014) & (df['date'].dt.month == 9)]
# filtered.loc[:, 'revenue'] = filtered.loc[:, 'item_cnt_day'] * filtered.loc[:, 'item_price']
#
# shop_revenue = filtered.groupby('shop_id').sum()
# res = shop_revenue.loc[:, 'revenue'].max()
# filtered = new_df.groupby('item_category_id')['revenue'].sum()
filtered = new_df.groupby('date')['item_cnt_day'].sum()

print(type(filtered))
days = pd.date_range(start="20141201", end="20141230", freq='D')
# filtered = pd.Series(filtered, index=filtered.loc[:, 'date'])
print(type(filtered))
print(filtered.values)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# fig, ax = plt.subplots()
#
# ax.plot_date(filtered.index.to_pydatetime(), filtered, 'v-')
#
# # ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=(1),
# #                                                 interval=1))
# ax.xaxis.set_minor_formatter(dates.DateFormatter('%d\n%m'))
# ax.xaxis.grid(True, which="minor")
# ax.yaxis.grid()
# ax.xaxis.set_major_locator(dates.DayLocator())
# ax.xaxis.set_major_formatter(dates.DateFormatter('\n\n%d\n%m'))
# plt.tight_layout()
plt.plot(filtered.index, filtered.values)
plt.xlabel("Day")
plt.ylabel("Num items")

plt.show()

print(np.var(filtered.values))
# plt.plot(days, filtered, kind)
# plt.ylabel('Num items')
# plt.xlabel('Day')
# plt.title("Daily revenue for shop_id = 25")
# plt.show()

# print(filtered)