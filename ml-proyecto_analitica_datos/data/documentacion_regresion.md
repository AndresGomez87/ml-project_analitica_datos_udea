# Documentación del Dataset — Regresión (Airbnb)

**Nombre del Dataset**

InsideAirbnb – Listings + Calendar (dataset preparado para regresión)

---

**Fuente (URL):** https://insideairbnb.com/get-the-data/

---

## Descripción general del problema

Con el aumento inminente de las rentas cortas en diferentes ciudades del mundo a través de la plataforma de Airbnb, resulta interesante y necesario analizar qué factores juegan un papel decisivo al momento de predecir el precio que va a tener una vivienda por medio de esta plataforma y así poder tomar decisiones que permitan optimizar y mejorar este aspecto turístico y sus repercusiones.

---

## Objetivo del análisis

Construir un modelo de regresión que estime el precio por noche de cada anuncio en función de las variables disponibles. Además se busca identificar las variables más influyentes y evaluar el desempeño del modelo.

---

## Variable objetivo (variable respuesta)

- `price`: precio por noche (numérica continua).

---

## Diccionario de variables (principales)

| Nombre | Descripción | Tipo |
|---|---|---|
| `listing_id` | Identificador único del anuncio | Categórica nominal / ID |
| `date` | Fecha del registro (si se usa calendar) | Categórica ordinal (fecha) |
| `available` | Disponibilidad en la fecha (t/f) | Categórica nominal |
| `price` | Precio por noche (p. ej. en USD) | Numérica continua |
| `host_id` | Identificador del anfitrión | Categórica nominal / ID |
| `neighbourhood` | Barrio o zona del anuncio | Categórica nominal |
| `latitude` | Latitud geográfica | Numérica continua |
| `longitude` | Longitud geográfica | Numérica continua |
| `room_type` | Tipo de alojamiento (Entire home/Private room/Shared) | Categórica nominal |
| `number_of_reviews` | Número total de reseñas | Numérica discreta |
| `reviews_per_month` | Reseñas por mes | Numérica continua |
| `availability_365` | Días disponibles en los últimos 365 días | Numérica discreta |
| `name` | Título del anuncio | Texto |
| `description` | Descripción del anuncio | Texto |

---

## Número de Observaciones y Variables

```
Número de observaciones (filas): 27051
Número de variables (columnas): 18
```

---

## Posibles hipótesis de estudio

1. **Hipótesis 1:** El precio por noche aumenta con el número de revisiones (`number_of_reviews`) y el tipo de habitación (`room_type`).
2. **Hipótesis 2:** Qué incidencia tiene la variable (`availability_365`) en la variable objetivo (precio).
3. **Hipótesis 3:** La ubicación (barrio/`neighbourhood`) tiene un efecto estadísticamente significativo en el precio, incluso después de controlar por tamaño y servicios.
4. **Hipótesis 4 *(temporal)*:** Los precios muestran estacionalidad semanal y anual (fines de semana y meses turísticos con precios más altos).
