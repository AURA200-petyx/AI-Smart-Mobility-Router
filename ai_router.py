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
        return abs(R * c)

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
    def process_natural_language_query(cls, message: str, lat: Optional[Any] = None, lon: Optional[Any] = None, action: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # ВЫВОДИМ В ЛОГИ РЕНДЕРА ВСЕ, ЧТО ПРИХОДИТ С ФРОНТЕНДА
        print(f"[DEBUG COORDS]: action={action}, raw_lat={lat} (type: {type(lat)}), raw_lon={lon} (type: {type(lon)}), message={message}")

        if action == "alternative":
            if cls._last_results:
                cls._last_index = (cls._last_index + 1) % len(cls._last_results)
            else:
                return {"success": False, "reply": "⚠️ Сначала выполните поиск объекта через ИИ-ассистент."}
        else:
            msg_clean = message.strip()
            
            system_prompt = (
                "You are an AI intent classifier for an urban navigation system. "
                "Classify the user input into EXACTLY one of these categories: "
                "cafe, pharmacy, supermarket, fuel, cinema. "
                "If it does not match any, output: unknown. "
                "Return ONLY the category name in lowercase, no quotes, no extra words."
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
                    max_tokens=150
                )
                
                msg_obj = completion.choices[0].message
                raw_content = getattr(msg_obj, "content", "") or ""
                if not raw_content and hasattr(msg_obj, "reasoning"):
                    raw_content = str(msg_obj.reasoning)
                
                raw_content = raw_content.strip().lower()
                valid_categories = ["cafe", "pharmacy", "supermarket", "fuel", "cinema"]
                for cat in valid_categories:
                    if cat in raw_content:
                        category = cat
                        break
            except Exception as e:
                return {
                    "success": False,
                    "reply": f"⚠️ Ошибка вызова Groq API: {str(e)}"
                }

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

            # Безопасное приведение координат
            try:
                f_lat = float(lat) if lat is not None else 53.9006
                f_lon = float(lon) if lon is not None else 27.5590
            except (ValueError, TypeError):
                f_lat, f_lon = 53.9006, 27.5590

            if abs(f_lat) < 1.0 or abs(f_lon) < 1.0:
                f_lat, f_lon = 53.9006, 27.5590

            print(f"[DEBUG COORDS]: final parsed f_lat={f_lat}, f_lon={f_lon}")

            try:
                headers = {"User-Agent": "AISmartUrbanRouter/10.0"}
                results = []
                
                with httpx.Client(timeout=6.0) as client:
                    # Ищем объекты в радиусе вокруг переданных координат
                    delta = 0.15
                    viewbox = f"{f_lon-delta},{f_lat-delta},{f_lon+delta},{f_lat+delta}"
                    url = f"https://nominatim.openstreetmap.org/search?q={target_info['tag']}&format=json&limit=25&lat={f_lat}&lon={f_lon}&viewbox={viewbox}&bounded=0"
                    resp = client.get(url, headers=headers)
                    results = resp.json()

                    if not results:
                        url_global = f"https://nominatim.openstreetmap.org/search?q={target_info['tag']}+Minsk&format=json&limit=25"
                        resp = client.get(url_global, headers=headers)
                        results = resp.json()

                    if results:
                        sorted_places = []
                        for place in results:
                            try:
                                p_lat = float(place["lat"])
                                p_lon = float(place["lon"])
                                dist = cls.calculate_distance(f_lat, f_lon, p_lat, p_lon)
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