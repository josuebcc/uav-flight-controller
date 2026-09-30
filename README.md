# uav-flight-controller

Repositorio asociado al Trabajo Fin de Máster sobre el diseño e implementación
de un controlador de vuelo.

## Entorno

Requiere Python 3.12. Desde la raíz del repositorio, en un terminal sin ROS cargado:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Para reproducir exactamente las versiones usadas: `pip install -r requirements-lock.txt`
seguido de `pip install -e . --no-deps`.

Comprobaciones:

```bash
pytest            # tests
ruff check .      # linter
ruff format .     # formateo
mypy              # tipos
```

## Estado

Proyecto en fase inicial. El código y la documentación se irán incorporando
conforme avance el trabajo.
