# import scrapy

# class IMDbReviewsSpider(scrapy.Spider):
#     name = "imdb_reviews"

#     def __init__(self, movie_ids=None, *args, **kwargs):
#         super(IMDbReviewsSpider, self).__init__(*args, **kwargs)
#         self.movie_ids = movie_ids.split(',') if movie_ids else []

#     def start_requests(self):
#         for movie_id in self.movie_ids:
#             url = f'https://www.imdb.com/title/{movie_id}/reviews'
#             yield scrapy.Request(url=url, callback=self.parse, meta={'movie_id': movie_id})

#     def parse(self, response):
#         try:
#             movie_id = response.meta['movie_id']
#             for review in response.css('div.lister-item-content div.text.show-more__control'):
#                 # Extract full review text
#                 full_review_text = " ".join(review.css('::text').getall())
#                 yield {'movie_id': movie_id, 'reviews': full_review_text.strip()}

#             # Follow pagination if available
#             next_page = response.css('div.load-more-data::attr(data-key)').get()
#             if next_page is not None:
#                 next_page_url = f'/title/{movie_id}/reviews/_ajax?ref_=undefined&paginationKey={next_page}'
#                 yield response.follow(next_page_url, self.parse, meta={'movie_id': movie_id})
#         except Exception as e:
#             self.logger.error(f"Error occurred while parsing: {str(e)}")



import scrapy

class IMDbReviewsSpider(scrapy.Spider):
    name = "imdb_reviews"

    def __init__(self, movie_ids=None, *args, **kwargs):
        super(IMDbReviewsSpider, self).__init__(*args, **kwargs)
        self.movie_ids = movie_ids.split(',') if movie_ids else []

    def start_requests(self):
        if not self.movie_ids:
            self.logger.error("No movie IDs provided. Please provide at least one movie ID.")
            return

        for movie_id in self.movie_ids:
            url = f'https://www.imdb.com/title/{movie_id}/reviews'
            yield scrapy.Request(url=url, callback=self.parse, meta={'movie_id': movie_id})

    def parse(self, response):
        try:
            movie_id = response.meta['movie_id']
            if 'No reviews yet.' in response.text:
                self.logger.warning(f"No reviews found for movie ID: {movie_id}")
                return

            for review in response.css('div.lister-item-content div.text.show-more__control'):
                # Extract full review text
                full_review_text = " ".join(review.css('::text').getall())
                yield {'movie_id': movie_id, 'reviews': full_review_text.strip()}

            # Follow pagination if available
            next_page = response.css('div.load-more-data::attr(data-key)').get()
            if next_page is not None:
                next_page_url = f'/title/{movie_id}/reviews/_ajax?ref_=undefined&paginationKey={next_page}'
                yield response.follow(next_page_url, self.parse, meta={'movie_id': movie_id})
        except Exception as e:
            self.logger.error(f"Error occurred while parsing: {str(e)}")
