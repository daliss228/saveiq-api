from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from supabase import create_client
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import joblib
import os

hoy = datetime.now()
primer_dia_mes_actual = datetime(hoy.year, hoy.month,1)
ultimo_dia_mes_anterior = (primer_dia_mes_actual - timedelta(days=1))
primer_dia_mes_anterior = datetime(ultimo_dia_mes_anterior.year, ultimo_dia_mes_anterior.month,1)

load_dotenv()

router = APIRouter()

# Supabase

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# Modelos

BASE_DIR = Path(__file__).resolve().parent.parent
modelo_kmeans = joblib.load(
    BASE_DIR / "models" / "kmeans_model.pkl"
)

modelo_scaler = joblib.load(
    BASE_DIR / "models" / "scaler.pkl"
)

# Perfiles

PERFILES = {
    0: "Ahorrador disciplinado",
    1: "Usuario estable",
    2: "Riesgo financiero alto"
}

@router.post("/predict/{user_id}")
def predict(user_id: str):

    try:

        # Verificar si existen movimientos

        gastos = (
            supabase
            .table("gastos")
            .select("id")
            .eq("usuario_id", user_id)
            .gte(
                "fecha",
                primer_dia_mes_anterior.isoformat()
            )
            .lt(
                "fecha",
                primer_dia_mes_actual.isoformat()
            )
            .execute()
        )

        ingresos = (
            supabase
            .table("ingresos")
            .select("id")
            .eq("usuario_id", user_id)
            .gte(
                "fecha",
                primer_dia_mes_anterior.isoformat()
            )
            .lt(
                "fecha",
                primer_dia_mes_actual.isoformat()
            )
            .execute()
        )

        usar_perfil = (
            len(gastos.data) == 0 and
            len(ingresos.data) == 0
        )

        # CASO 1
        # Datos onboarding perfil usuario

        if usar_perfil:

            perfil = (
                supabase
                .table("perfiles")
                .select("*")
                .eq("id", user_id)
                .single()
                .execute()
            )

            if not perfil.data:
                raise HTTPException(
                    status_code=404,
                    detail="Perfil no encontrado"
                )

            ingreso_mensual = float(
                perfil.data["ingreso_mensual"]
            )

            prestamos_activos = float(
                perfil.data["prestamos_activos"]
            )

            ahorros = float(
                perfil.data["ahorros"]
            )

            gasto_esencial = float(
                perfil.data["gasto_esencial"]
            )

            gasto_discrecional = float(
                perfil.data["gasto_discrecional"]
            )

        # CASO 2
        # Datos tablas

        else:

            # Tabla Ingresos

            ingresos_mes = (
                supabase
                .table("ingresos")
                .select("monto")
                .eq("usuario_id", user_id)
                .execute()
            )

            ingreso_mensual = sum(
                float(i["monto"])
                for i in ingresos_mes.data
            )

            # Tabla Prestamos

            prestamos = (
                supabase
                .table("prestamos")
                .select("monto")
                .eq("pagado", False)
                .eq("usuario_id", user_id)
                .execute()
            )

            prestamos_activos = sum(
                float(p["monto"])
                for p in prestamos.data
            )

            # Separar gastos esenciales y discrecionales

            gastos_data = (
                supabase
                .table("gastos")
                .select("""
                    monto,
                    etiquetas (
                        es_esencial
                    )
                """)
                .eq("usuario_id", user_id)
                .execute()
            )

            gasto_esencial = 0
            gasto_discrecional = 0

            for gasto in gastos_data.data:

                monto = float(
                    gasto["monto"]
                )

                es_esencial = gasto[
                    "etiquetas"
                ]["es_esencial"]

                if es_esencial:
                    gasto_esencial += monto
                else:
                    gasto_discrecional += monto

            # Tabla Objetivos

            objetivos = (
                supabase
                .table("objetivos")
                .select("cantidad_ahorrado")
                .eq("usuario_id", user_id)
                .eq("logrado", False)
                .execute()
            )

            ahorros = sum(
                float(o["cantidad_ahorrado"])
                for o in objetivos.data
            )

        # Evitar división por cero

        if ingreso_mensual <= 0:

            raise HTTPException(
                status_code=400,
                detail="Ingreso mensual inválido"
            )

        # Variables del modelo

        expense_ratio = (
            gasto_esencial +
            gasto_discrecional
        ) / ingreso_mensual

        debt_ratio = (
            prestamos_activos
            /
            ingreso_mensual
        )

        savings_ratio = (
            ahorros
            /
            ingreso_mensual
        )

        discretionary_ratio = (
            gasto_discrecional
            /
            ingreso_mensual
        )

        # DataFrame

        nuevo_usuario = pd.DataFrame([{
            "expense_ratio": expense_ratio,
            "debt_ratio": debt_ratio,
            "savings_ratio": savings_ratio,
            "discretionary_ratio": discretionary_ratio
        }])


        # Escalado

        nuevo_scaled = modelo_scaler.transform(
            nuevo_usuario
        )

        # Predicción

        cluster = int(modelo_kmeans.predict(nuevo_scaled)[0])

        perfil_cluster = PERFILES.get(
            cluster,
            "Desconocido"
        )

        ahorro_estimado = round(
            ahorros,
            2
        )

        # Guardar resultado

        supabase.table(
            "predicciones"
        ).insert({

            "usuario_id":
                user_id,

            "cluster":
                cluster,

            "perfil":
                perfil_cluster

        }).execute()

        return {

            "cluster":
                cluster,

            "perfil":
                perfil_cluster,

            "expense_ratio":
                round(expense_ratio, 4),

            "debt_ratio":
                round(debt_ratio, 4),

            "savings_ratio":
                round(savings_ratio, 4),

            "discretionary_ratio":
                round(discretionary_ratio, 4)

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )