import requests
import ssl
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from urllib.request import urlopen
import time
import logging
from typing import List, Set, Dict, Optional
from dataclasses import dataclass, field
import re
import os
import csv
import hashlib
from datetime import datetime
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScrapingConfig:
    """Configuration pour le scraping éthique"""
    delay: float = 1.0  # Délai entre les requêtes (secondes)
    max_depth: Optional[int] = 3  # Profondeur maximale pour le deep scraping
    respect_robots_txt: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0"
    timeout: int = 10
    max_pages: Optional[int] = 100  # Limite du nombre de pages à scraper
    verify_ssl: bool = True
    content_selector: Optional[str] = None
    exclude_selectors: List[str] = field(default_factory=lambda: ['script', 'style', 'nav', 'footer', 'header'])
    keep_selectors: List[str] = field(default_factory=list)  # Sélecteurs CSS à garder malgré exclude_selectors
    extract_links_from_content_only: bool = False #Extraire uniquement les liens depuis la zone ciblée
    link_selectors: List[str] = field(default_factory=list)
    allowed_url_patterns: List[str] = field(default_factory=list)
    content_types: List[str] = field(default_factory=lambda: ["text"]) # Type de contenu que l'on récupère
    img_allow: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "webp", "gif"])
    save_urls: bool = False
    save_urls_directory: str = "scraped_csv"


@dataclass
class ScrapedContent:
    """Contenu scrapé d'une page"""
    url: str
    title: Optional[str] = None
    text: str = ""
    images: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    depth: int = 0
    status_code: int = 0


class BasicScraper:
    """
    Scraper éthique respectant les bonnes pratiques:
    - Respect du robots.txt
    - Délai entre les requêtes
    - User-agent identifiable
    - Gestion des erreurs
    - Limitation du nombre de requêtes
    """

    def __init__(self, config: Optional[ScrapingConfig] = None):
        self.config = config or ScrapingConfig()
        self.session = requests.Session()
        self.session.verify = config.verify_ssl
        self.session.headers.update({'User-Agent': self.config.user_agent})
        self.robots_parsers: Dict[str, RobotFileParser] = {}
        self.visited_urls: Set[str] = set()
        self.last_request_time: Dict[str, float] = {}

    def _get_domain(self, url: str) -> str:
        """Extrait le domaine d'une URL"""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _can_fetch(self, url: str) -> bool:
        """Vérifie si l'URL peut être scrapée selon robots.txt"""
        if not self.config.respect_robots_txt:
            return True

        domain = self._get_domain(url)

        if domain not in self.robots_parsers:
            robots_url = urljoin(domain, '/robots.txt')
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
                self.robots_parsers[domain] = rp
                logger.info(f"robots.txt chargé pour {domain}")
            except Exception as e:
                logger.warning(f"Impossible de charger robots.txt pour {domain}: {e}")
                # En cas d'erreur, on autorise par défaut (comportement permissif)
                return True

        return self.robots_parsers[domain].can_fetch(self.config.user_agent, url)

    def _respect_delay(self, url: str):
        """Respecte le délai entre les requêtes pour un même domaine"""
        domain = self._get_domain(url)

        random_delay = random.uniform(self.config.delay * 0.8, self.config.delay * 1.5)

        if domain in self.last_request_time:
            elapsed = time.time() - self.last_request_time[domain]
            if elapsed < random_delay:
                sleep_time = random_delay - elapsed
                logger.info(f"Attente de {sleep_time:.2f}s avant la prochaine requête")
                time.sleep(sleep_time)

        self.last_request_time[domain] = time.time()

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extrait toutes les images de la page"""
        raw_images = set()

        allowed_exts = self.config.img_allow

        for img in soup.find_all('img', src=True):
            src = img['src'].strip()

            # Gestion des images inline base64
            if src.startswith("data:image/"):
                raw_images.add(src)
                continue

            # Convertir en URL absolue
            absolute_url = urljoin(base_url, src)

            path = urlparse(absolute_url).path
            ext = path.split('.')[-1].lower()
            if ext in allowed_exts:
                raw_images.add(absolute_url)

        return sorted(raw_images)

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extrait et filtre les liens d'une page selon les patterns autorisés"""
        raw_links = set()

        for tag in soup.find_all('a', href=True):
            href = tag['href'].strip()

            # Ignore les liens non-HTML
            if href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
                continue

            # Convertit en URL absolue
            absolute_url = urljoin(base_url, href)
            parsed = urlparse(absolute_url)

            # Nettoyage de base : enlever le fragment
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                clean_url += f"?{parsed.query}"

            raw_links.add(clean_url)

        # --- Application des patterns de filtrage si configurés ---
        if self.config.allowed_url_patterns:
            filtered_links = set()
            for link in raw_links:
                for pattern in self.config.allowed_url_patterns:
                    if re.search(pattern, link):
                        filtered_links.add(link)
                        break  # pas besoin de tester les autres patterns
            links = sorted(filtered_links)
        else:
            links = sorted(raw_links)

        logger.debug(f"{len(links)} liens extraits après filtrage sur {len(raw_links)} bruts")
        return links


    def scrape_page(self, url: str) -> Optional[ScrapedContent]:
        """
        Scrape une seule page

        Args:
            url: URL de la page à scraper

        Returns:
            ScrapedContent ou None en cas d'erreur
        """
        # Vérifications éthiques
        if not self._can_fetch(url):
            logger.warning(f"Scraping non autorisé par robots.txt: {url}")
            return None

        if url in self.visited_urls:
            logger.debug(f"URL déjà visitée: {url}")
            return None

        # Respect du délai
        self._respect_delay(url)

        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()

            self.visited_urls.add(url)

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extraction du contenu
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else None

            # Ciblage du contenu principal si un sélecteur est défini
            content_soup = soup
            if self.config.content_selector:
                target = soup.select_one(self.config.content_selector)
                if target:
                    content_soup = target
                    logger.debug(f"Contenu ciblé avec '{self.config.content_selector}'")
                else:
                    logger.warning(f"Sélecteur '{self.config.content_selector}' non trouvé, utilisation de la page complète")

            text = ""
            if any(ct in self.config.content_types for ct in ["text", "text+images"]):
                # Extraction du texte (en retirant les éléments indésirables avec exceptions)
                # D'abord, marquer les éléments à garder
                elements_to_keep = []
                for keep_selector in self.config.keep_selectors:
                    elements_to_keep.extend(content_soup.select(keep_selector))

                # Ensuite, supprimer les éléments exclus sauf ceux marqués à garder
                for exclude_selector in self.config.exclude_selectors:
                    for element in content_soup.select(exclude_selector):
                        if element not in elements_to_keep:
                            element.decompose()

                text = content_soup.get_text(separator=' ', strip=True)

            # --- Construction de la zone combinée pour les liens ---
            link_soups = []

            # zone principale si extract_links_from_content_only=True
            if self.config.extract_links_from_content_only:
                link_soups.append(content_soup)
            else:
                link_soups.append(soup)

            # zones additionnelles spécifiées via link_selectors
            for link_selector in self.config.link_selectors:
                elements = soup.select(link_selector)
                if elements:
                    link_soups.extend(elements)
                    logger.debug(f"Liens ajoutés depuis '{link_selector}' ({len(elements)} éléments)")

            # Fusion des HTMLs pour extraire les liens d’un seul coup
            combined_html = "".join(str(el) for el in link_soups)
            combined_soup = BeautifulSoup(combined_html, 'html.parser')
            links = self._extract_links(combined_soup, url)

            # Extraction des images
            images = []

            if any(ct in self.config.content_types for ct in ["images", "text+images"]):
                images = self._extract_images(content_soup, url)

            content = ScrapedContent(
                url=url,
                title=title_text,
                text=text,
                links=links,
                images=images,
                status_code=response.status_code
            )

            logger.info(f"✓ Scrapé: {url} (titre: {title_text})")
            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Erreur lors du scraping de {url}: {e}")
            return None

    def scrape_multiple(self, urls: List[str]) -> List[ScrapedContent]:
        """
        Scrape plusieurs pages

        Args:
            urls: Liste d'URLs à scraper

        Returns:
            Liste des contenus scrapés
        """
        results = []
        for url in urls:
            if self.config.max_pages is not None and len(self.visited_urls) >= self.config.max_pages:
                logger.warning(f"Limite de {self.config.max_pages} pages atteinte")
                break

            content = self.scrape_page(url)
            if content:
                results.append(content)

        return results

    def deep_scrape(self, start_url: str, same_domain_only: bool = True) -> List[ScrapedContent]:
        """
        Scraping en profondeur suivant les liens

        Args:
            start_url: URL de départ
            same_domain_only: Si True, ne scrape que les liens du même domaine

        Returns:
            Liste des contenus scrapés
        """
        results = []
        to_visit = [(start_url, 0)]  # (url, depth)
        start_domain = self._get_domain(start_url)

        if self.config.save_urls:
            self._init_csv_files(self.config.save_urls_directory)

        while to_visit:

            # Vérification de la limite de pages
            if self.config.max_pages is not None and len(self.visited_urls) >= self.config.max_pages:
                logger.warning(f"Limite de {self.config.max_pages} pages atteinte")
                break

            current_url, depth = to_visit.pop(0)

            # Vérification de la profondeur (None = illimité)
            if self.config.max_depth is not None and depth > self.config.max_depth:
                continue

            # Vérification du domaine
            if same_domain_only and self._get_domain(current_url) != start_domain:
                continue

            # Scraping de la page
            content = self.scrape_page(current_url)

            # Enregistrement dans les csv
            self.to_csv(content, to_save=self.config.save_urls)

            if content:
                content.depth = depth
                results.append(content)

                # Ajout des nouveaux liens à explorer (si pas de limite de profondeur ou si sous la limite)
                if self.config.max_depth is None or depth < self.config.max_depth:
                    for link in content.links:
                        if link not in self.visited_urls and (link, depth + 1) not in to_visit:
                            to_visit.append((link, depth + 1))

                logger.info(f"Profondeur {depth}: {len(to_visit)} liens en attente")

        return results, to_visit

    def reset(self):
        """Réinitialise l'état du scraper"""
        self.visited_urls.clear()
        self.last_request_time.clear()
        logger.info("Scraper réinitialisé")

    def _init_csv_files(self, output_dir="scraped_csv"):
        """Initialise les fichiers CSV avec un timestamp unique."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_stamp = hashlib.md5(timestamp.encode()).hexdigest()[:8]

        os.makedirs(output_dir, exist_ok=True)

        self.csv_text_path = os.path.join(output_dir, f"text_{hash_stamp}.csv")
        self.csv_page_path = os.path.join(output_dir, f"pages_{hash_stamp}.csv")
        self.csv_img_path = os.path.join(output_dir, f"images_{hash_stamp}.csv")

        # Créer les fichiers vides avec headers
        with open(self.csv_text_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["url", "title", "text"])

        with open(self.csv_page_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["url", "title", "status_code"])

        with open(self.csv_img_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_url"])

        # Set pour éviter doublons d'images
        self._seen_images: Set[str] = set()

    def to_csv(self, content: ScrapedContent, to_save: bool = True):
        """
        Sauvegarde le contenu scrapé dans les CSV correspondants.

        Args:
            content: ScrapedContent
            to_save: bool, si True on écrit dans les fichiers
        """
        if not to_save or content is None:
            return

        # Texte
        with open(self.csv_text_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([content.url, content.title or "", content.text])

        # Page
        with open(self.csv_page_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([content.url, content.title or "", content.status_code])

        # Images (éviter doublons)
        new_images = [img for img in content.images if img not in self._seen_images]
        if new_images:
            with open(self.csv_img_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for img_url in new_images:
                    writer.writerow([img_url])
                    self._seen_images.add(img_url)