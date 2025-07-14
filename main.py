"""
Uso del pipeline de procesamiento.
"""

import os
from src.pipeline_procesamiento import PipelineProcesamiento


def main():
    """
    Función principal para ejecutar el pipeline.
    """
    try:
        # Configurar rutas
        ruta_excel = "data/raw/ESTIM_papa_2005-2025 (1).xls"
        carpeta_atmosfericos = "data/raw/DatosAtmosfericos"
        carpeta_salida = "data/processed"

        # Verificar que las rutas existen antes de continuar
        if not os.path.exists(ruta_excel):
            print(f"❌ Error: No se encontró el archivo Excel: {ruta_excel}")
            print("   Verifica que la ruta sea correcta.")
            return

        if not os.path.exists(carpeta_atmosfericos):
            print(f"❌ Error: No se encontró la carpeta de datos atmosféricos: {carpeta_atmosfericos}")
            print("   Verifica que la ruta sea correcta.")
            return

        print(f"🚀 Iniciando pipeline de procesamiento de datos...")
        print(f"📂 Archivo Excel: {ruta_excel}")
        print(f"📁 Carpeta atmosféricos: {carpeta_atmosfericos}")
        print(f"💾 Carpeta de salida: {carpeta_salida}")
        print(f"{'=' * 60}")

        # Crear instancia del pipeline
        pipeline = PipelineProcesamiento(
            ruta_excel_papa=ruta_excel,
            carpeta_datos_atmosfericos=carpeta_atmosfericos,
            carpeta_salida=carpeta_salida,
            log_level="INFO"
        )

        # Ejecutar pipeline completo
        ruta_final = pipeline.ejecutar_pipeline_completo()

        # Mostrar resultados
        print(f"\n{'=' * 60}")
        print("✅ PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print(f"{'=' * 60}")
        print(f"📄 Archivo final guardado en: {ruta_final}")


        # Verificar que el archivo final existe
        if os.path.exists(ruta_final):
            tamano_archivo = os.path.getsize(ruta_final) / 1024 / 1024  # MB
            print(f"📏 Tamaño del archivo final: {tamano_archivo:.2f} MB")

        print(f"\n🎉 ¡Pipeline ejecutado con éxito!")

    except FileNotFoundError as e:
        print(f"❌ Error: Archivo no encontrado - {e}")
        print("   Verifica que todas las rutas de archivos sean correctas.")

    except Exception as e:
        print(f"❌ Error inesperado durante el procesamiento: {e}")
        print("   Revisa los logs para más detalles.")

        # Mostrar información adicional para debugging
        print(f"\n🔍 Información de debugging:")
        print(f"  • Tipo de error: {type(e).__name__}")
        print(f"  • Mensaje: {str(e)}")


if __name__ == "__main__":
    main()