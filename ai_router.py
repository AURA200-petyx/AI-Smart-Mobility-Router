import httpx
import math
import itertools
from typing import Dict, List, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AIRouter:
    TRAIN_DATA = [
        ("где поесть перекусить кафе ресторан кофе пицца обед ужин столовая бургер еда", "cafe"),
        ("аптека купить лекарства болит голова таблетки рецепт фармацевт аптечный", "pharmacy"),
        ("магазин супермаркет продукты купить еду гипермаркет молоко хлеб", "supermarket"),
        ("заправка бензин топливо азс заправиться машина дизель колонка", "fuel"),
        ("кинотеатр фильм посмотреть кино сеанс билеты премьера", "cinema")
    ]
    
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    corpus = [item[0] for item in TRAIN_DATA]
    labels = [item[1] for item in TRAIN_DATA]
    X_train = vectorizer.fit_transform(corpus)

    _last_results: List[Dict[str, Any]] = []
    _last_index: int = 0
    _last_category_name: str = ""

    @classmethod
    def calculate_distance(cls, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        
        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @classmethod
    def process_natural_language_query(cls, message: str, lat: Optional[float], lon: Optional[float], action: Optional[str] = None) -> Dict[str, Any]:
        if action == "alternative":
            if cls._last_results:
                cls._last_index = (cls._last_index + 1) % len(cls._last_results)
            else:
                return {"success": False, "reply": "⚠️ Сначала выполните поиск объекта через запрос к ИИ."}
        else:
            msg_clean = message.strip().lower()
            user_vector = cls.vectorizer.transform([msg_clean])
            similarities = cosine_similarity(user_vector, cls.X_train)[0]
            best_idx = similarities.argmax()
            confidence = float(similarities[best_idx])
            
            if confidence < 0.12:
                return {
                    "success": False,
                    "reply": f"🤖 ИИ проанализировал запрос («<i>{message}</i>») и не нашел категории. Спросите: <i>«Где выпить кофе?»</i> или <i>«Нужна аптека»</i>."
                }
            
            category = cls.labels[best_idx]
            
            osm_queries = {
                "cafe": {"tag": "cafe", "name": "Кафе / Ресторан", "desc": "Заведение общественного питания."},
                "pharmacy": {"tag": "pharmacy", "name": "Аптека", "desc": "Аптечный пункт медицинских товаров."},
                "supermarket": {"tag": "supermarket", "name": "Супермаркет", "desc": "Продуктовый магазин."},
                "fuel": {"tag": "fuel", "name": "Автозаправка (АЗС)", "desc": "Топливная станция."},
                "cinema": {"tag": "cinema", "name": "Кинотеатр", "desc": "Центр показа фильмов."}
            }
            
            target_info = osm_queries.get(category, {"tag": "amenity", "name": "Объект", "desc": "Городской объект."})
            cls._last_category_name = target_info["name"]

            if not lat or not lon or abs(lat) < 1.0 or abs(lon) < 1.0:
                lat, lon = 53.9006, 27.5590

            try:
                delta = 0.05
                viewbox = f"{lon-delta},{lat+delta},{lon+delta},{lat-delta}"
                url = f"https://nominatim.openstreetmap.org/search?q={target_info['tag']}&format=json&limit=15&lat={lat}&lon={lon}&viewbox={viewbox}&bounded=1"
                headers = {"User-Agent": "AISmartUrbanRouter/10.0"}
                
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(url, headers=headers)
                    results = resp.json()
                    
                    if not results:
                        url_fallback = f"https://nominatim.openstreetmap.org/search?q={target_info['tag']}&format=json&limit=15&lat={lat}&lon={lon}"
                        resp = client.get(url_fallback, headers=headers)
                        results = resp.json()

                    if results:
                        sorted_places = []
                        for place in results:
                            p_lat = float(place["lat"])
                            p_lon = float(place["lon"])
                            dist = cls.calculate_distance(lat, lon, p_lat, p_lon)
                            sorted_places.append({
                                "place": place,
                                "dist": dist,
                                "desc": target_info["desc"]
                            })
                        
                        sorted_places.sort(key=lambda x: x["dist"])
                        cls._last_results = sorted_places
                        cls._last_index = 0
            except Exception:
                pass

        if cls._last_results:
            item = cls._last_results[cls._last_index]
            place = item["place"]
            p_lat = float(place["lat"])
            p_lon = float(place["lon"])
            dist = int(item["dist"])
            
            full_name = place.get("display_name", cls._last_category_name)
            short_name = full_name.split(",")[0]
            
            reply_text = (
                f"🤖 <b>AI Smart Agent (Вариант {cls._last_index + 1} из {len(cls._last_results)}):</b><br>"
                f"📍 <b>Найдено:</b> {short_name}<br>"
                f"📏 <b>Прямое расстояние:</b> ~{dist} метров<br>"
                f"📝 <b>Описание:</b> {item['desc']}<br><br>"
                f"<button onclick=\"previewAiPoint({p_lat}, {p_lon}, '{short_name}')\" style='padding:4px 8px; background:#2980b9; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold;'>👁️ Посмотреть</button> "
                f"<button onclick=\"selectAiPoint({p_lat}, {p_lon}, '{short_name}')\" style='padding:4px 8px; background:#27ae60; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold;'>➕ Добавить в маршрут</button> "
                f"<button onclick=\"requestAlternative()\" style='padding:4px 8px; background:#e67e22; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold;'>🔄 Другой</button>"
            )

            return {
                "success": True,
                "target_name": short_name,
                "lat": p_lat,
                "lon": p_lon,
                "reply": reply_text
            }

        return {
            "success": False,
            "reply": "⚠️ Не удалось найти подходящие объекты поблизости."
        }

    @classmethod
    def analyze_and_route(cls, grid: list, points: List[Any], optimization_mode: str = "sequence", mode: Optional[str] = None) -> Dict[str, Any]:
        active_mode = optimization_mode if optimization_mode != "sequence" else (mode or "sequence")
        
        if not points or len(points) < 2:
            return {"status": "error", "message": "Недостаточно точек для расчета"}

        formatted_points = []
        for p in points:
            if isinstance(p, dict):
                formatted_points.append({
                    "id": p.get("id"),
                    "lat": float(p.get("lat", 0)), 
                    "lon": float(p.get("lon", 0)),
                    "name": p.get("name", "")
                })
            elif hasattr(p, "lat") and hasattr(p, "lon"):
                formatted_points.append({
                    "id": getattr(p, "id", None),
                    "lat": float(p.lat), 
                    "lon": float(p.lon),
                    "name": getattr(p, "name", "")
                })

        ordered_points = list(formatted_points)
        
        if active_mode == "shortest" and len(formatted_points) > 2:
            if len(formatted_points) > 9:
                return {
                    "status": "error",
                    "message": "⚠️ Превышен лимит для оптимизации (максимум 9 точек). Уменьшите количество точек или выберите режим «По порядку»."
                }
            
            start = formatted_points[0]
            targets = formatted_points[1:]
            
            best_route = None
            min_distance = float('inf')
            
            for perm in itertools.permutations(targets):
                candidate = [start] + list(perm)
                cand_dist = 0.0
                for i in range(len(candidate) - 1):
                    cand_dist += cls.calculate_distance(
                        candidate[i]["lat"], candidate[i]["lon"],
                        candidate[i+1]["lat"], candidate[i+1]["lon"]
                    )
                
                if cand_dist < min_distance:
                    min_distance = cand_dist
                    best_route = candidate
                    
            if best_route:
                ordered_points = best_route

        total_dist = 0.0
        for i in range(len(ordered_points) - 1):
            total_dist += cls.calculate_distance(
                ordered_points[i]["lat"], ordered_points[i]["lon"],
                ordered_points[i+1]["lat"], ordered_points[i+1]["lon"]
            )

        speed_kmh = 30.0
        dist_factor = 1.30

        final_dist_m = int(total_dist * dist_factor)
        duration_sec = int(final_dist_m / (speed_kmh * 1000 / 3600))

        return {
            "status": "success",
            "mode": active_mode,
            "optimized_waypoints": ordered_points,
            "estimated_dist_m": final_dist_m,
            "estimated_duration_sec": max(15, duration_sec)
        }