import requests
from bs4 import BeautifulSoup
import json
import os

# Base URL of the courses
BASE_URL = "https://courses.analyticsvidhya.com"

# URL format for paginated pages
URL_TEMPLATE = "https://courses.analyticsvidhya.com/collections/courses?page={page}"

# Output file path for saving scraped data
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/courses.json")


def scrape_courses():
    # Initialize list to hold course data
    courses = []

    # Loop through all pages (1 to 9)
    for page in range(1, 10):
        print(f"Scraping page {page}...")
        response = requests.get(URL_TEMPLATE.format(page=page))

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to fetch page {page}. Status code: {response.status_code}")
            continue

        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Locate course containers
        course_items = soup.find_all("li", class_="products__list-item")
        print(f"Found {len(course_items)} course containers on page {page}.")

        # Extract details for each course
        for item in course_items:
            # Extract course link
            link_tag = item.find("a", class_="course-card")
            course_link = link_tag.get("href", "#") if link_tag else "#"
            if not course_link.startswith("http"):
                course_link = f"{BASE_URL}{course_link}"

            # Extract course title
            title_tag = link_tag.find("h3") if link_tag else None
            title = title_tag.text.strip() if title_tag else "No Title"

            # Extract course image
            image_tag = (
                link_tag.find("img", class_="course-card__img") if link_tag else None
            )
            image_url = (
                image_tag.get("src", "No Image URL") if image_tag else "No Image URL"
            )

            # Extract course description
            lesson_tag = (
                link_tag.find("span", class_="course-card__lesson-count")
                if link_tag
                else None
            )
            description = lesson_tag.text.strip() if lesson_tag else "No Description"

            # Add the extracted details to the list
            courses.append(
                {
                    "title": title,
                    "description": description,
                    "image_url": image_url,
                    "course_link": course_link,
                }
            )

    # Ensure the directory for the output file exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Save the course data to a JSON file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(courses, f, indent=4)

    print(f"Scraping completed. Data saved to {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    scrape_courses()
