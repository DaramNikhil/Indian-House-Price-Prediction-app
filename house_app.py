from pydantic import BaseModel

class Item(BaseModel):
    price_per_sqft: float
    area: float
    bedRoom: int
    bathroom: int
    additionalRoom: int
    plot_area_sqft: float
    average_rating: float

    