import os
import math
import httpx
from typing import Dict, List, Optional, Any
from openai import OpenAI

class GroqAIRouter:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )

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
    def analyze_and_route(cls, grid: Optional[List[Any]] = None, points: Optional[List[Dict[str, Any]]] = None, optimization_mode: Optional[str] = "sequence", mode: Optional[str] = "sequence", **kwargs) -> Dict[str, Any]:
        pts = points or []
        if not pts:
            return {"status": "error", "message": "Нет точек для построения маршрута", "points": []}
        
        ordered_points = pts.copy()
        
        if optimization_mode in ["nearest", "tsp", "optimized", "true", "1"] and len(pts) > 2:
            start = pts[0]
            unvisited = pts[1:]
            ordered = [start]
            
            while unvisited:
                current = ordered[-1]
                cur_lat = float(current.get("lat", current.get("latitude", 0)))
                cur_lon = float(current.get("lon", current.get("longitude", 0)))
                
                next_point = min(unvisited, key=lambda p: cls.calculate_distance(
                    cur_lat, cur_lon,
                    float(p.get("lat", p.get("latitude", 0))), float(p.get("lon", p.get("longitude", 0)))
                ))
                ordered.append(next_point)
                unvisited.remove(next_point)
            ordered_points = ordered

        return {
            "status": "success",
            "points": ordered_points,
            "message": "Маршрут успешно оптимизирован"
        }

    @classmethod
    def process_natural_language_query(cls, message: str, lat: Optional[float] = None, lon: Optional[float] = None, action: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if action == "alternative":
            if cls._last_results:
                cls._last_index = (cls._last_index + 1) % len(cls._last_results)
            else:
                return {"success": False, "reply": "⚠️ Сначала выполните поиск объекта через ИИ-ассистент."}
        else:
            msg_clean = message.strip()
            
            system_prompt = (
                "You are the core semantic intent classifier for an autonomous geospatial navigation system. "
                "Map the user's natural language query into EXACTLY one of these categories: "
                "cafe, pharmacy, supermarket, fuel, cinema. "
                "If the query does not clearly match any of these categories, output 'unknown'. "
                "Return ONLY the category name in lowercase, with zero punctuation, quotes, or additional text."
            )

            category = "unknown"
            try:
                completion = cls.client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": msg_clean}
                    ],
                    temperature=0.0,
                    max_tokens=10
                )
                
                raw_content = completion.choices[0].message.content.strip().lower()
                
                valid_categories = ["cafe", "pharmacy", "supermarket", "fuel", "cinema"]
                for cat in valid_categories:
                    if cat in raw_content:
                        category = cat
                        break
            except Exception as e:
                return {
                    "success": False,
                    "reply": f"⚠️ Ошибка обращения к ИИ-сервису Groq: {str(e)}"
                }

            # СТРОГОЕ ПРАВИЛО: если ИИ не понял — возвращаем ошибку, никакого фоллбека на кафе!
            if category == "unknown":
                return {
                    "success": False,
                    "reply": f"🤖 ИИ-ассистент проанализировал запрос («<i>{message}</i>») и не смог сопоставить его с объектами карты. Попробуйте спросить иначе."
                }

            osm_queries = {
                "cafe": {"tag": "cafe", "name": "Кафе / Ресторан", "desc": "Заведение общественного питания."},
                "pharmacy": {"tag": "pharmacy", "name": "Аптека", "desc": "Аптечный пункт."},
                "supermarket": {"tag": "supermarket", "name": "Супермаркет", "desc": "Продуктовый магазин."},
                "fuel": {"tag": "fuel", "name": "АЗС", "desc": "Топливная станция."},
                "cinema": {"tag": "cinema", "name": "Кинотеатр", "desc": "Кинотеатр."}
            }
            
            target_info = osm_queries.get(category, {"tag": "amenity", "name": "Объект", "desc": "Городской объект."})
            cls._last_category_name = target_info["name"]

            if not lat or not lon or abs(lat) < 1.0 or abs(lon) < 1.0:
                lat, lon = 53.9006, 27.5590

            try:
                headers = {"User-Agent": "AISmartUrbanRouter/10.0"}
                results = []
                
                with httpx.Client(timeout=6.0) as client:
                    url = f"https://nominatim.openstreetmap.org/search?q={target_info['tag']}&format=json&limit=10&lat={lat}&lon={lon}"
                    resp = client.get(url, headers=headers)
                    results = resp.json()
                    
                    if not results:
                        url_global = f"https://nominatim.openstreetmap.org/search?q={target_info['tag']}+Minsk&format=json&limit=10"
                        resp = client.get(url_global, headers=headers)
                        results = resp.json()

                    if results:
                        sorted_places = []
                        for place in results:
                            try:
                                p_lat = float(place["lat"])
                                p_lon = float(place["lon"])
                                dist = cls.calculate_distance(lat, lon, p_lat, p_lon)
                                sorted_places.append({
                                    "place": place,
                                    "dist": dist,
                                    "desc": target_info["desc"]
                                })
                            except (KeyError, ValueError):
                                continue
                        
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
            
            raw_name = place.get("display_name", "").split(",")[0]
            if raw_name.lower() in ["cafe", "pharmacy", "supermarket", "fuel", "cinema", "amenity"]:
                short_name = cls._last_category_name
            else:
                short_name = raw_name or cls._last_category_name
            
            reply_text = (
                f"🤖 <b>Groq AI Agent (Вариант {cls._last_index + 1} из {len(cls._last_results)}):</b><br>"
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
            "reply": "⚠️ Не удалось найти подходящие объекты поблизости. Попробуйте еще раз."
        }