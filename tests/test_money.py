import pytest

from agents.money import find_inr_amounts, format_money, parse_amount, parse_inr


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Apple iPhone 15 (128 GB) at ₹1,29,999", [129999]),
        ("boAt Airdopes 141 @ ₹1,099 (MRP ₹4,490)", [1099, 4490]),
        ("Samsung 43 inch 4K TV just Rs. 24,990", [24990]),
        ("Redmi 13C 5G Rs 9,499 only", [9499]),
        ("Prestige induction cooktop INR 1,899", [1899]),
        ("Noise smartwatch ₹49,999/- deal", [49999]),
        ("Fastrack watch 1,295/- only today", [1295]),
        ("Mivi Play speaker @264 https://amzn.to/4abc", [264]),
        ("Lenovo IdeaPad Slim 3 at 10,999 after coupon", [10999]),
        ("Philips trimmer ₹1435 / 65% off", [1435]),
        ("Current Price: ₹2,149\nHighest Price: ₹3,999", [2149, 3999]),
        ("Upto 85% off on Puma shoes", []),
        ("Bank offer: 10% off on HDFC cards, 5000mAh battery, 128GB storage", []),
        ("Wildcraft backpack for 2 years warranty", []),
        ("Bajaj mixer grinder M.R.P.: ₹5,200 Deal: ₹2,799", [5200, 2799]),
        ("Rs.399.50 for a 2 pack of Duracell batteries", [399.5]),
        ("Sony WH-1000XM5 ₹ 26,990 + ₹2,000 bank discount", [26990, 2000]),
        ("Western grouping also works: ₹129,999", [129999]),
        ("Price: 799 for Boult earbuds", [799]),
        ("Flat ₹200 cashback on orders above ₹999 in Amazon", [200, 999]),
    ],
)
def test_parse_inr(text, expected):
    assert parse_inr(text) == expected


def test_find_inr_amounts_keeps_positions_and_markers():
    text = "MRP ₹4,490 now @1,099"
    amounts = find_inr_amounts(text)
    assert [(a.value, a.marker) for a in amounts] == [(4490, "₹"), (1099, "@")]
    assert text[amounts[0].start : amounts[0].end] == "₹4,490"


def test_parse_inr_empty():
    assert parse_inr("") == []
    assert parse_inr(None) == []


@pytest.mark.parametrize(
    "amount, currency, expected",
    [
        (129999, "INR", "₹1,29,999"),
        (49999.0, "INR", "₹49,999"),
        (999, "INR", "₹999"),
        (1000, "INR", "₹1,000"),
        (10000000, "INR", "₹1,00,00,000"),
        (1299.5, "INR", "₹1,299.50"),
        (-500, "INR", "-₹500"),
        (0, "INR", "₹0"),
        (1299.99, "USD", "$1,299.99"),
        (None, "INR", "-"),
    ],
)
def test_format_money(amount, currency, expected):
    assert format_money(amount, currency) == expected


@pytest.mark.parametrize(
    "text, expected",
    [("₹1,29,999", 129999), ("1299.00", 1299), ("$45", 45), ("", None), (None, None)],
)
def test_parse_amount(text, expected):
    assert parse_amount(text) == expected
