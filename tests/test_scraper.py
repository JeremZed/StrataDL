import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from bs4 import BeautifulSoup
import requests
from urllib.robotparser import RobotFileParser
import time
import os
import csv
from datetime import datetime

from stratadl.core.ingestion.scraper import BasicScraper, ScrapingConfig, ScrapedContent


class TestScrapingConfig:
    """Tests pour la classe ScrapingConfig"""

    def test_default_values(self):
        config = ScrapingConfig()
        assert config.delay == 1.0
        assert config.max_depth == 3
        assert config.respect_robots_txt is True
        assert config.timeout == 10
        assert config.max_pages == 100
        assert config.verify_ssl is True
        assert config.content_selector is None
        assert config.extract_links_from_content_only is False
        assert config.save_urls is False
        assert "text" in config.content_types
        assert "script" in config.exclude_selectors

    def test_custom_values(self):
        config = ScrapingConfig(
            delay=2.0,
            max_depth=5,
            respect_robots_txt=False,
            max_pages=50
        )
        assert config.delay == 2.0
        assert config.max_depth == 5
        assert config.respect_robots_txt is False
        assert config.max_pages == 50


class TestScrapedContent:
    """Tests pour la classe ScrapedContent"""

    def test_creation(self):
        content = ScrapedContent(
            url="https://example.com",
            title="Test Title",
            text="Test content",
            images=["img1.jpg", "img2.png"],
            links=["https://example.com/page1"],
            depth=1,
            status_code=200
        )
        assert content.url == "https://example.com"
        assert content.title == "Test Title"
        assert content.text == "Test content"
        assert len(content.images) == 2
        assert len(content.links) == 1
        assert content.depth == 1
        assert content.status_code == 200


class TestBasicScraper:
    """Tests pour la classe BasicScraper"""

    @pytest.fixture
    def scraper(self):
        config = ScrapingConfig(delay=0.1, respect_robots_txt=False)
        return BasicScraper(config)

    @pytest.fixture
    def mock_response(self):
        response = Mock()
        response.status_code = 200
        response.content = b"""
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Content</h1>
                <p>This is test content.</p>
                <a href="/page1">Link 1</a>
                <a href="https://example.com/page2">Link 2</a>
                <img src="/image1.jpg" />
                <img src="https://example.com/image2.png" />
                <script>console.log('test');</script>
            </body>
        </html>
        """
        response.raise_for_status = Mock()
        return response

    # Tests de la méthode _get_domain
    def test_get_domain(self, scraper):
        assert scraper._get_domain("https://example.com/page") == "https://example.com"
        assert scraper._get_domain("http://test.org/path/to/page") == "http://test.org"
        assert scraper._get_domain("https://sub.domain.com:8080/page") == "https://sub.domain.com:8080"

    # Tests de la méthode _can_fetch
    def test_can_fetch_respect_robots_false(self, scraper):
        scraper.config.respect_robots_txt = False
        assert scraper._can_fetch("https://example.com/any-page") is True

    @patch('urllib.robotparser.RobotFileParser')
    def test_can_fetch_with_robots_txt(self, mock_robot_parser):
        config = ScrapingConfig(respect_robots_txt=True)
        scraper = BasicScraper(config)

        mock_rp = Mock()
        mock_rp.can_fetch.return_value = True
        mock_robot_parser.return_value = mock_rp

        result = scraper._can_fetch("https://example.com/page")
        assert result is True

    @patch('urllib.robotparser.RobotFileParser')
    def test_can_fetch_robots_txt_error(self, mock_robot_parser):
        config = ScrapingConfig(respect_robots_txt=True)
        scraper = BasicScraper(config)

        mock_rp = Mock()
        mock_rp.read.side_effect = Exception("Connection error")
        mock_robot_parser.return_value = mock_rp

        # En cas d'erreur, doit autoriser par défaut
        result = scraper._can_fetch("https://example.com/page")
        assert result is True

    # Tests de la méthode _respect_delay
    def test_respect_delay_first_request(self, scraper):
        url = "https://example.com/page"
        start_time = time.time()
        scraper._respect_delay(url)
        elapsed = time.time() - start_time

        # Premier appel, pas de délai
        assert elapsed < 0.1

    def test_respect_delay_subsequent_request(self, scraper):
        scraper.config.delay = 0.5
        url = "https://example.com/page"

        scraper._respect_delay(url)
        start_time = time.time()
        scraper._respect_delay(url)
        elapsed = time.time() - start_time

        # Doit attendre environ 0.5 secondes (avec variation aléatoire)
        assert elapsed >= 0.3  # 0.5 * 0.8 minimum

    # Tests de la méthode _extract_images
    def test_extract_images_basic(self, scraper):
        html = """
        <html>
            <body>
                <img src="image1.jpg" />
                <img src="/path/image2.png" />
                <img src="https://example.com/image3.webp" />
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        images = scraper._extract_images(soup, "https://example.com")

        assert len(images) == 3
        assert "https://example.com/image1.jpg" in images
        assert "https://example.com/path/image2.png" in images
        assert "https://example.com/image3.webp" in images

    def test_extract_images_base64(self, scraper):
        html = """
        <html>
            <body>
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" />
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        images = scraper._extract_images(soup, "https://example.com")

        assert len(images) == 1
        assert images[0].startswith("data:image/")

    def test_extract_images_filters_extensions(self, scraper):
        scraper.config.img_allow = ["jpg", "png"]
        html = """
        <html>
            <body>
                <img src="image1.jpg" />
                <img src="image2.gif" />
                <img src="image3.png" />
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        images = scraper._extract_images(soup, "https://example.com")

        assert len(images) == 2
        assert any("jpg" in img for img in images)
        assert any("png" in img for img in images)
        assert not any("gif" in img for img in images)

    # Tests de la méthode _extract_links
    def test_extract_links_basic(self, scraper):
        html = """
        <html>
            <body>
                <a href="/page1">Link 1</a>
                <a href="https://example.com/page2">Link 2</a>
                <a href="page3">Link 3</a>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = scraper._extract_links(soup, "https://example.com")

        assert len(links) == 3
        assert "https://example.com/page1" in links
        assert "https://example.com/page2" in links
        assert "https://example.com/page3" in links

    def test_extract_links_filters_non_http(self, scraper):
        html = """
        <html>
            <body>
                <a href="https://example.com/page1">Valid Link</a>
                <a href="mailto:test@example.com">Email</a>
                <a href="tel:+123456789">Phone</a>
                <a href="javascript:void(0)">JS Link</a>
                <a href="#section">Anchor</a>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = scraper._extract_links(soup, "https://example.com")

        assert len(links) == 1
        assert "https://example.com/page1" in links

    def test_extract_links_with_patterns(self, scraper):
        scraper.config.allowed_url_patterns = [r"/blog/", r"/products/"]
        html = """
        <html>
            <body>
                <a href="/blog/post1">Blog Post</a>
                <a href="/products/item1">Product</a>
                <a href="/about">About</a>
                <a href="/contact">Contact</a>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = scraper._extract_links(soup, "https://example.com")

        assert len(links) == 2
        assert "https://example.com/blog/post1" in links
        assert "https://example.com/products/item1" in links

    # Tests de la méthode scrape_page
    @patch('requests.Session.get')
    def test_scrape_page_success(self, mock_get, scraper, mock_response):
        mock_get.return_value = mock_response

        content = scraper.scrape_page("https://example.com")

        assert content is not None
        assert content.url == "https://example.com"
        assert content.title == "Test Page"
        assert "Main Content" in content.text
        assert "This is test content" in content.text
        assert len(content.links) > 0
        assert content.status_code == 200

    @patch('requests.Session.get')
    def test_scrape_page_already_visited(self, mock_get, scraper):
        url = "https://example.com"
        scraper.visited_urls.add(url)

        content = scraper.scrape_page(url)

        assert content is None
        mock_get.assert_not_called()

    @patch('requests.Session.get')
    def test_scrape_page_request_error(self, mock_get, scraper):
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        content = scraper.scrape_page("https://example.com")

        assert content is None

    @patch('requests.Session.get')
    def test_scrape_page_with_content_selector(self, mock_get, scraper):
        scraper.config.content_selector = ".main-content"
        response = Mock()
        response.status_code = 200
        response.content = b"""
        <html>
            <head><title>Test</title></head>
            <body>
                <nav>Navigation</nav>
                <div class="main-content">
                    <p>Main text</p>
                </div>
                <footer>Footer</footer>
            </body>
        </html>
        """
        response.raise_for_status = Mock()
        mock_get.return_value = response

        content = scraper.scrape_page("https://example.com")

        assert "Main text" in content.text
        assert "Navigation" not in content.text
        assert "Footer" not in content.text

    @patch('requests.Session.get')
    def test_scrape_page_extract_images(self, mock_get, scraper):
        scraper.config.content_types = ["images"]
        response = Mock()
        response.status_code = 200
        response.content = b"""
        <html>
            <body>
                <img src="image1.jpg" />
                <img src="image2.png" />
            </body>
        </html>
        """
        response.raise_for_status = Mock()
        mock_get.return_value = response

        content = scraper.scrape_page("https://example.com")

        assert len(content.images) == 2
        assert content.text == ""

    @patch('requests.Session.get')
    def test_scrape_page_text_and_images(self, mock_get, scraper):
        scraper.config.content_types = ["text+images"]
        response = Mock()
        response.status_code = 200
        response.content = b"""
        <html>
            <body>
                <p>Some text</p>
                <img src="image.jpg" />
            </body>
        </html>
        """
        response.raise_for_status = Mock()
        mock_get.return_value = response

        content = scraper.scrape_page("https://example.com")

        assert len(content.images) == 1
        assert "Some text" in content.text

    @patch('requests.Session.get')
    def test_scrape_page_keep_selectors(self, mock_get, scraper):
        scraper.config.exclude_selectors = ['nav', 'footer']
        scraper.config.keep_selectors = ['nav.important']
        response = Mock()
        response.status_code = 200
        response.content = b"""
        <html>
            <body>
                <nav class="important">Important Nav</nav>
                <nav>Regular Nav</nav>
                <p>Content</p>
                <footer>Footer</footer>
            </body>
        </html>
        """
        response.raise_for_status = Mock()
        mock_get.return_value = response

        content = scraper.scrape_page("https://example.com")

        assert "Important Nav" in content.text
        assert "Regular Nav" not in content.text
        assert "Footer" not in content.text

    @patch('requests.Session.get')
    def test_scrape_page_extract_links_from_content_only(self, mock_get, scraper):
        scraper.config.extract_links_from_content_only = True
        scraper.config.content_selector = ".content"
        response = Mock()
        response.status_code = 200
        response.content = b"""
        <html>
            <body>
                <nav><a href="/nav-link">Nav Link</a></nav>
                <div class="content">
                    <a href="/content-link">Content Link</a>
                </div>
            </body>
        </html>
        """
        response.raise_for_status = Mock()
        mock_get.return_value = response

        content = scraper.scrape_page("https://example.com")

        assert "https://example.com/content-link" in content.links
        assert "https://example.com/nav-link" not in content.links

    @patch('requests.Session.get')
    def test_scrape_page_with_link_selectors(self, mock_get, scraper):
        scraper.config.link_selectors = ['.sidebar']
        response = Mock()
        response.status_code = 200
        response.content = b"""
        <html>
            <body>
                <div class="content">
                    <a href="/page1">Link 1</a>
                </div>
                <div class="sidebar">
                    <a href="/page2">Link 2</a>
                </div>
            </body>
        </html>
        """
        response.raise_for_status = Mock()
        mock_get.return_value = response

        content = scraper.scrape_page("https://example.com")

        assert "https://example.com/page1" in content.links
        assert "https://example.com/page2" in content.links

    # Tests de la méthode scrape_multiple
    @patch('requests.Session.get')
    def test_scrape_multiple(self, mock_get, scraper, mock_response):
        mock_get.return_value = mock_response
        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3"
        ]

        results = scraper.scrape_multiple(urls)

        assert len(results) == 3
        assert all(isinstance(r, ScrapedContent) for r in results)

    @patch('requests.Session.get')
    def test_scrape_multiple_respects_max_pages(self, mock_get, scraper, mock_response):
        scraper.config.max_pages = 2
        mock_get.return_value = mock_response
        urls = [f"https://example.com/page{i}" for i in range(5)]

        results = scraper.scrape_multiple(urls)

        assert len(results) == 2

    # Tests de la méthode deep_scrape
    @patch('requests.Session.get')
    def test_deep_scrape_basic(self, mock_get, scraper):
        # Mock des réponses pour différentes pages
        response1 = Mock()
        response1.status_code = 200
        response1.content = b"""
        <html>
            <head><title>Page 1</title></head>
            <body>
                <a href="https://example.com/page2">Link to Page 2</a>
            </body>
        </html>
        """
        response1.raise_for_status = Mock()

        response2 = Mock()
        response2.status_code = 200
        response2.content = b"""
        <html>
            <head><title>Page 2</title></head>
            <body>
                <p>Content of page 2</p>
            </body>
        </html>
        """
        response2.raise_for_status = Mock()

        mock_get.side_effect = [response1, response2]

        results, remaining = scraper.deep_scrape("https://example.com/page1")

        assert len(results) >= 1
        assert any(r.title == "Page 1" for r in results)

    @patch('requests.Session.get')
    def test_deep_scrape_respects_max_depth(self, mock_get, scraper):
        scraper.config.max_depth = 1

        response = Mock()
        response.status_code = 200
        response.content = b"""
        <html>
            <head><title>Test</title></head>
            <body>
                <a href="https://example.com/page2">Link</a>
            </body>
        </html>
        """
        response.raise_for_status = Mock()
        mock_get.return_value = response

        results, remaining = scraper.deep_scrape("https://example.com")

        # Devrait limiter la profondeur
        assert all(r.depth <= 1 for r in results)

    @patch('requests.Session.get')
    def test_deep_scrape_same_domain_only(self, mock_get, scraper):
        response = Mock()
        response.status_code = 200
        response.content = b"""
        <html>
            <body>
                <a href="https://example.com/page2">Same Domain</a>
                <a href="https://other.com/page">Other Domain</a>
            </body>
        </html>
        """
        response.raise_for_status = Mock()
        mock_get.return_value = response

        results, remaining = scraper.deep_scrape("https://example.com", same_domain_only=True)

        # Vérifier que seul le même domaine est visité
        assert all("example.com" in r.url for r in results)

    @patch('requests.Session.get')
    def test_deep_scrape_respects_max_pages(self, mock_get, scraper):
        scraper.config.max_pages = 2

        response = Mock()
        response.status_code = 200
        response.content = b"""
        <html>
            <body>
                <a href="https://example.com/page2">Link 2</a>
                <a href="https://example.com/page3">Link 3</a>
                <a href="https://example.com/page4">Link 4</a>
            </body>
        </html>
        """
        response.raise_for_status = Mock()
        mock_get.return_value = response

        results, remaining = scraper.deep_scrape("https://example.com")

        assert len(results) <= 2

    # Tests de la méthode reset
    def test_reset(self, scraper):
        scraper.visited_urls.add("https://example.com")
        scraper.last_request_time["https://example.com"] = time.time()

        scraper.reset()

        assert len(scraper.visited_urls) == 0
        assert len(scraper.last_request_time) == 0

    # Tests des méthodes CSV
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_init_csv_files(self, mock_file, mock_makedirs, scraper):
        scraper._init_csv_files("test_dir")

        assert mock_makedirs.called
        assert scraper.csv_text_path.startswith("test_dir/text_")
        assert scraper.csv_page_path.startswith("test_dir/pages_")
        assert scraper.csv_img_path.startswith("test_dir/images_")

    @patch('builtins.open', new_callable=mock_open)
    def test_to_csv_disabled(self, mock_file, scraper):
        content = ScrapedContent(
            url="https://example.com",
            title="Test",
            text="Content"
        )

        scraper.to_csv(content, to_save=False)

        mock_file.assert_not_called()

    @patch('builtins.open', new_callable=mock_open)
    def test_to_csv_enabled(self, mock_file, scraper):
        scraper._init_csv_files()
        content = ScrapedContent(
            url="https://example.com",
            title="Test",
            text="Content",
            images=["img1.jpg"],
            status_code=200
        )

        scraper.to_csv(content, to_save=True)

        assert mock_file.called

    @patch('builtins.open', new_callable=mock_open)
    def test_to_csv_avoids_duplicate_images(self, mock_file, scraper):
        scraper._init_csv_files()
        scraper._seen_images = set()

        content1 = ScrapedContent(
            url="https://example.com/page1",
            images=["img1.jpg", "img2.jpg"]
        )
        content2 = ScrapedContent(
            url="https://example.com/page2",
            images=["img2.jpg", "img3.jpg"]  # img2.jpg est un doublon
        )

        scraper.to_csv(content1, to_save=True)
        scraper.to_csv(content2, to_save=True)

        assert "img1.jpg" in scraper._seen_images
        assert "img2.jpg" in scraper._seen_images
        assert "img3.jpg" in scraper._seen_images

    @patch('requests.Session.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_deep_scrape_with_csv_save(self, mock_makedirs, mock_file, mock_get, scraper):
        scraper.config.save_urls = True

        response = Mock()
        response.status_code = 200
        response.content = b"<html><head><title>Test</title></head><body></body></html>"
        response.raise_for_status = Mock()
        mock_get.return_value = response

        results, remaining = scraper.deep_scrape("https://example.com")

        assert mock_file.called


# Tests d'intégration
class TestIntegration:
    """Tests d'intégration pour vérifier le comportement global"""

    @patch('requests.Session.get')
    def test_full_scraping_workflow(self, mock_get):
        # Configuration
        config = ScrapingConfig(
            delay=0.1,
            max_depth=2,
            max_pages=5,
            respect_robots_txt=False
        )
        scraper = BasicScraper(config)

        # Mock response
        response = Mock()
        response.status_code = 200
        response.content = b"""
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Content</h1>
                <p>Test text</p>
                <a href="/page2">Link</a>
                <img src="image.jpg" />
            </body>
        </html>
        """
        response.raise_for_status = Mock()
        mock_get.return_value = response

        # Exécution
        results, remaining = scraper.deep_scrape("https://example.com")

        # Assertions
        assert len(results) > 0
        assert all(isinstance(r, ScrapedContent) for r in results)
        assert all(r.title is not None for r in results)