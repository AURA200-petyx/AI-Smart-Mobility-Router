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
    def process_natural_language_query(cls, message: str, lat: Optional[float], lon: Optional[float], action: Optional[str] = None) -> Dict[str, Any]:
        if action == "alternative":
            if cls._last_results:
                cls._last_index = (cls._last_index + 1) % len(cls._last_results)
            else:
                return {"success": False, "reply": "⚠️ Сначала выполните поиск объекта через ИИ-ассистент."}
        else:
            msg_clean = message.strip()
            
            # Строгий и понятный промпт для языковой модели
            system_prompt = (
                "You are a strict JSON-like semantic classifier for a mapping agent. "
                "Your task is to read the user request and output ONLY one of the exact words: "
                "cafe, pharmacy, supermarket, fuel, cinema. "
                "If no category matches, output: unknown. "
                "Do not write explanations, punctuation, or formatting."
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
                    max_tokens=15
                )
                
                raw_content = completion.choices[0].message.content.strip().lower()
                
                # Интеллектуальный поиск ключевой категории среди ответа модели
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

            if category == "unknown":
                return {
                    "success": False,
                    "reply": f"🤖 ИИ-ассистент проанализировал запрос («<i>{message}</i>») и не смог сопоставить его с объектами карты. Попробуйте спросить иначе."
                }

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
            "reply": "⚠️ Не удалось найти подходящие объекты поблизости."
        }