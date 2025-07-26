#!/usr/bin/env python3
"""
Sistema de Análisis de Resultados ESRGAN
Genera gráficos de violín y tablas de resumen para artículo
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ESRGANResultsAnalyzer:
    """Analizador de resultados para artículo científico"""
    
    def __init__(self, validation_dir="Validation"):
        """
        Inicializa el analizador
        
        Args:
            validation_dir: Directorio con todos los resultados de validación
        """
        self.validation_dir = Path(validation_dir)
        self.metrics = ['psnr', 'ssim', 'ms_ssim', 'mse', 'perceptual_index']
        
        # Configuración de modelos y sus parámetros arquitectónicos
        self.model_configs = {
            '64to128': {'scale': 2, 'params': {'num_features': 32, 'trunk_size': 11, 'growth_channel': 32}},
            '64to256': {'scale': 4, 'params': {'num_features': 32, 'trunk_size': 11, 'growth_channel': 32}},
            '64to1024': {'scale': 16, 'params': {'num_features': 32, 'trunk_size': 11, 'growth_channel': 32}},
            '128to256': {'scale': 2, 'params': {'num_features': 32, 'trunk_size': 11, 'growth_channel': 32}},
            '128to512': {'scale': 4, 'params': {'num_features': 32, 'trunk_size': 11, 'growth_channel': 32}},
            '128to1024': {'scale': 8, 'params': {'num_features': 64, 'trunk_size': 23, 'growth_channel': 64}},  # Problemas gradiente
            '256to512': {'scale': 2, 'params': {'num_features': 32, 'trunk_size': 11, 'growth_channel': 32}},
            '256to1024': {'scale': 4, 'params': {'num_features': 40, 'trunk_size': 11, 'growth_channel': 40}},  # Problemas gradiente
            '512to1024': {'scale': 2, 'params': {'num_features': 16, 'trunk_size': 6, 'growth_channel': 16}}    # Optimizado memoria
        }
        
    def load_all_metrics(self):
        """Carga todas las métricas de todos los modelos"""
        all_data = []
        
        # Buscar todos los archivos de métricas
        for model_dir in self.validation_dir.glob("*_metrics"):
            model_name = model_dir.name.replace("_metrics", "")
            base_model = self._extract_base_model(model_name)
            
            # Buscar archivos CSV de métricas
            for csv_file in model_dir.glob("*_metrics_*.csv"):
                # Determinar tipo (color/grayscale) y variante del modelo
                if "color" in csv_file.name:
                    image_type = "color"
                elif "grayscale" in csv_file.name:
                    image_type = "grayscale"
                else:
                    continue
                
                # Extraer variante del modelo del nombre del archivo
                model_variant = csv_file.stem.replace("_metrics_color", "").replace("_metrics_grayscale", "")
                
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Agregar información del modelo
                    df['model_name'] = model_variant
                    df['base_model'] = base_model
                    df['image_type'] = image_type
                    df['scale_factor'] = self.model_configs.get(base_model, {}).get('scale', 1)
                    df['target_resolution'] = int(base_model.split('to')[1])
                    df['input_resolution'] = int(base_model.split('to')[0])
                    
                    all_data.append(df)
                    print(f"✅ Cargado: {model_variant} ({image_type}) - {len(df)} imágenes")
                    
                except Exception as e:
                    print(f"❌ Error cargando {csv_file}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n📊 Total de datos cargados: {len(combined_df)} registros")
            return combined_df
        else:
            print("❌ No se encontraron datos de métricas")
            return None
    
    def load_all_timing(self):
        """Carga todos los datos de timing"""
        all_timing = []
        
        # Buscar archivos de timing
        for timing_file in self.validation_dir.glob("*/timing_results/*_timing.csv"):
            try:
                df = pd.read_csv(timing_file)
                
                # Extraer información del nombre del archivo
                filename = timing_file.stem
                parts = filename.split('_')
                
                # Encontrar el dispositivo (cpu o gpu)
                device = 'unknown'
                for part in parts:
                    if part in ['cpu', 'gpu']:
                        device = part
                        break
                
                # Extraer nombre del modelo (todo excepto device y 'timing')
                model_parts = [p for p in parts if p not in ['cpu', 'gpu', 'timing']]
                model_name = '_'.join(model_parts)
                base_model = self._extract_base_model(model_name)
                
                df['model_name'] = model_name
                df['base_model'] = base_model
                df['device'] = device
                df['scale_factor'] = self.model_configs.get(base_model, {}).get('scale', 1)
                
                all_timing.append(df)
                print(f"✅ Timing cargado: {model_name} ({device})")
                
            except Exception as e:
                print(f"❌ Error cargando timing {timing_file}: {e}")
        
        if all_timing:
            combined_timing = pd.concat(all_timing, ignore_index=True)
            print(f"\n⏱️  Total timing data: {len(combined_timing)} registros")
            return combined_timing
        else:
            print("❌ No se encontraron datos de timing")
            return None
    
    def _extract_base_model(self, model_name):
        """Extrae el modelo base del nombre completo"""
        # Buscar patrón XtoY en el nombre
        import re
        pattern = r'(\d+to\d+)'
        match = re.search(pattern, model_name)
        if match:
            return match.group(1)
        return model_name
    
    def create_violin_plot_1024_targets(self, metrics_df, output_path="violin_plot_1024_targets.png"):
        """
        Crea gráfico de violín para modelos que van a resolución 1024x1024
        
        Args:
            metrics_df: DataFrame con todas las métricas
            output_path: Ruta para guardar el gráfico
        """
        # Configurar fuente Computer Modern y tamaños más grandes
        plt.rcParams.update({
            'font.family': ['serif'],
            'font.serif': ['Computer Modern Roman', 'Times', 'DejaVu Serif'],
            'font.size': 16,           # Tamaño base aumentado
            'axes.titlesize': 18,      # Título principal
            'axes.labelsize': 16,      # Etiquetas de ejes
            'xtick.labelsize': 16,     # Etiquetas del eje X
            'ytick.labelsize': 16,     # Etiquetas del eje Y
            'legend.fontsize': 16,     # Leyenda
            'figure.titlesize': 22     # Título de figura
        })
        
        # Filtrar modelos que van a 1024
        target_1024_models = ['64to1024', '128to1024', '256to1024', '512to1024']
        df_1024 = metrics_df[
            (metrics_df['base_model'].isin(target_1024_models)) & 
            (metrics_df['image_type'] == 'color')  # Solo usar datos de color
        ].copy()
        
        if df_1024.empty:
            print("❌ No se encontraron datos para modelos que van a 1024")
            return
        
        # Configurar estilo
        sns.set_style("whitegrid")
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # Nombres de métricas en español
        metric_names = {
            'psnr': 'PSNR (dB)',
            'ssim': 'SSIM',
            'ms_ssim': 'MS-SSIM',
            'mse': 'Error Cuadrático Medio'
        }
        
        # 1. Crear gráfico principal con 4 métricas (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        # Crear violín para las 4 métricas principales
        metrics_to_plot = ['psnr', 'ssim', 'ms_ssim', 'mse']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric not in df_1024.columns:
                continue
            
            ax = axes[i]
            
            # Crear el violin plot
            violin_parts = ax.violinplot(
                [df_1024[df_1024['base_model'] == model][metric].values 
                 for model in target_1024_models],
                positions=range(len(target_1024_models)),
                showmeans=True,
                showmedians=True,
                showextrema=True
            )
            
            # Personalizar colores
            for pc, color in zip(violin_parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            # Configurar ejes
            ax.set_xticks(range(len(target_1024_models)))
            ax.set_xticklabels([f'{model}\n(×{self.model_configs[model]["scale"]})' 
                              for model in target_1024_models], rotation=45)
            
            # Título y labels en español
            metric_display = metric_names[metric]
            if metric == 'mse':
                ax.set_ylabel(f'{metric_display} (menor es mejor)')
            else:
                ax.set_ylabel(f'{metric_display} (mayor es mejor)')
                
            ax.set_title(f'Distribución de {metric_display} por Factor de Escala', fontweight='bold', pad=20)
            ax.set_xlabel('Modelo (Factor de Escala)')
            ax.grid(True, alpha=0.3)
            
            # Agregar estadísticas en el gráfico
            stats_text = []
            for j, model in enumerate(target_1024_models):
                model_data = df_1024[df_1024['base_model'] == model][metric]
                mean_val = model_data.mean()
                std_val = model_data.std()
                stats_text.append(f'{model}: μ={mean_val:.4f}±{std_val:.4f}')
            
            # Agregar texto de estadísticas
            stats_str = '\n'.join(stats_text)
            ax.text(0.02, 0.98, stats_str, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"🎻 Gráfico de violín principal guardado: {output_path}")
        
        # 2. Crear gráfico separado para índice perceptual
        perceptual_path = output_path.replace('.png', '_perceptual.png')
        
        if 'perceptual_index' in df_1024.columns:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Crear el violin plot para índice perceptual
            violin_parts = ax.violinplot(
                [df_1024[df_1024['base_model'] == model]['perceptual_index'].values 
                 for model in target_1024_models],
                positions=range(len(target_1024_models)),
                showmeans=True,
                showmedians=True,
                showextrema=True
            )
            
            # Personalizar colores
            for pc, color in zip(violin_parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            # Configurar ejes
            ax.set_xticks(range(len(target_1024_models)))
            ax.set_xticklabels([f'{model}\n(×{self.model_configs[model]["scale"]})' 
                              for model in target_1024_models], rotation=45)
            
            ax.set_ylabel('Índice Perceptual (menor es mejor)')
            ax.set_title('Distribución del Índice Perceptual por Factor de Escala', fontweight='bold', pad=20)
            ax.set_xlabel('Modelo (Factor de Escala)')
            ax.grid(True, alpha=0.3)
            
            # Agregar estadísticas
            stats_text = []
            for j, model in enumerate(target_1024_models):
                model_data = df_1024[df_1024['base_model'] == model]['perceptual_index']
                mean_val = model_data.mean()
                std_val = model_data.std()
                stats_text.append(f'{model}: μ={mean_val:.4f}±{std_val:.4f}')
            
            stats_str = '\n'.join(stats_text)
            ax.text(0.02, 0.98, stats_str, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(perceptual_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"🎻 Gráfico de índice perceptual guardado: {perceptual_path}")
        
        # Crear tabla estadística
        self._create_statistics_table_1024(df_1024, target_1024_models, 
                                          output_path.replace('.png', '_stats.csv'))
        
        # Restaurar configuración por defecto
        plt.rcdefaults()

    def _create_statistics_table_1024(self, df_1024, models, output_path):
        """Crea tabla de estadísticas para los modelos 1024"""
        stats_data = []
        
        for model in models:
            model_data = df_1024[df_1024['base_model'] == model]
            
            row = {
                'Model': model,
                'Scale': f"×{self.model_configs[model]['scale']}",
                'Samples': len(model_data)
            }
            
            for metric in self.metrics:
                if metric in model_data.columns:
                    values = model_data[metric]
                    row[f'{metric}_mean'] = values.mean()
                    row[f'{metric}_std'] = values.std()
                    row[f'{metric}_median'] = values.median()
            
            stats_data.append(row)
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(output_path, index=False)
        print(f"📊 Estadísticas guardadas: {output_path}")
    
    def create_summary_tables(self, metrics_df, timing_df, output_dir="summary_tables"):
        """
        Crea tablas de resumen para LaTeX
        
        Args:
            metrics_df: DataFrame con métricas
            timing_df: DataFrame con timing
            output_dir: Directorio para guardar tablas
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Tabla de métricas promedio por modelo (solo color)
        metrics_color = metrics_df[metrics_df['image_type'] == 'color']
        
        summary_metrics = []
        for model_name in metrics_color['model_name'].unique():
            model_data = metrics_color[metrics_color['model_name'] == model_name]
            base_model = model_data['base_model'].iloc[0]
            
            row = {
                'Model': model_name,
                'Scale': f"×{self.model_configs.get(base_model, {}).get('scale', '?')}",
                'Input_Res': f"{model_data['input_resolution'].iloc[0]}×{model_data['input_resolution'].iloc[0]}",
                'Target_Res': f"{model_data['target_resolution'].iloc[0]}×{model_data['target_resolution'].iloc[0]}",
                'Samples': len(model_data)
            }
            
            # Parámetros arquitectónicos
            params = self.model_configs.get(base_model, {}).get('params', {})
            row['Num_Features'] = params.get('num_features', '?')
            row['Trunk_Size'] = params.get('trunk_size', '?')
            row['Growth_Channel'] = params.get('growth_channel', '?')
            
            # Métricas
            for metric in self.metrics:
                if metric in model_data.columns:
                    row[f'{metric.upper()}_mean'] = f"{model_data[metric].mean():.4f}"
                    row[f'{metric.upper()}_std'] = f"{model_data[metric].std():.4f}"
            
            summary_metrics.append(row)
        
        metrics_summary_df = pd.DataFrame(summary_metrics)
        metrics_summary_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
        
        # 2. Tabla de timing promedio
        if timing_df is not None:
            timing_summary = []
            
            for model_name in timing_df['model_name'].unique():
                model_timing = timing_df[timing_df['model_name'] == model_name]
                
                for device in ['gpu', 'cpu']:
                    device_data = model_timing[model_timing['device'] == device]
                    if len(device_data) == 0:
                        continue
                    
                    row = {
                        'Model': model_name,
                        'Device': device.upper(),
                        'Samples': len(device_data),
                        'Mean_Time_ms': f"{device_data['mean_time_ms'].mean():.2f}",
                        'Std_Time_ms': f"{device_data['mean_time_ms'].std():.2f}",
                        'FPS_mean': f"{device_data['fps'].mean():.2f}",
                        'Memory_MB': f"{device_data['memory_increase_mb'].mean():.1f}"
                    }
                    timing_summary.append(row)
            
            timing_summary_df = pd.DataFrame(timing_summary)
            timing_summary_df.to_csv(f"{output_dir}/timing_summary.csv", index=False)
        
        # 3. Tabla LaTeX-ready combinada
        self._create_latex_table(metrics_summary_df, timing_summary_df if timing_df is not None else None, 
                                output_dir)
        
        print(f"📋 Tablas de resumen guardadas en: {output_dir}")
    
    def _create_latex_table(self, metrics_df, timing_df, output_dir):
        """Crea tabla formateada para LaTeX"""
        
        # Tabla de métricas principal
        latex_content = """
% Tabla de Métricas de Modelos ESRGAN
\\begin{table*}[htbp]
\\centering
\\caption{Performance Metrics of ESRGAN Models for Histopathology Super-Resolution}
\\label{tab:esrgan_metrics}
\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}
\\hline
\\textbf{Model} & \\textbf{Scale} & \\textbf{Resolution} & \\textbf{Architecture} & \\textbf{PSNR} & \\textbf{SSIM} & \\textbf{MS-SSIM} & \\textbf{Perceptual} & \\textbf{Samples} \\\\
& & (Input→Target) & (NF, TS, GC) & (dB) & & & Index & \\\\
\\hline
"""
        
        for _, row in metrics_df.iterrows():
            model_name = row['Model'].replace('_', '\\_')
            resolution = f"{row['Input_Res']}→{row['Target_Res']}"
            architecture = f"({row['Num_Features']}, {row['Trunk_Size']}, {row['Growth_Channel']})"
            
            # Marcar modelos optimizados
            if 'Optimized' in row['Model'] or row['Num_Features'] != 32:
                model_name += "*"
            
            latex_content += f"{model_name} & {row['Scale']} & {resolution} & {architecture} & "
            latex_content += f"{row.get('PSNR_mean', 'N/A')} & {row.get('SSIM_mean', 'N/A')} & "
            latex_content += f"{row.get('MS_SSIM_mean', 'N/A')} & {row.get('PERCEPTUAL_INDEX_mean', 'N/A')} & "
            latex_content += f"{row['Samples']} \\\\\n"
        
        latex_content += """\\hline
\\end{tabular}
\\begin{tablenotes}
\\item[*] Optimized architectures with modified parameters for memory efficiency
\\item NF: Number of Features, TS: Trunk Size, GC: Growth Channel
\\end{tablenotes}
\\end{table*}

"""
        
        # Tabla de timing si está disponible
        if timing_df is not None:
            latex_content += """
% Tabla de Timing
\\begin{table}[htbp]
\\centering
\\caption{Inference Time Comparison: GPU vs CPU}
\\label{tab:inference_timing}
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Model} & \\textbf{Device} & \\textbf{Time (ms)} & \\textbf{FPS} & \\textbf{Memory (MB)} \\\\
\\hline
"""
            
            for _, row in timing_df.iterrows():
                model_name = row['Model'].replace('_', '\\_')
                latex_content += f"{model_name} & {row['Device']} & {row['Mean_Time_ms']} & "
                latex_content += f"{row['FPS_mean']} & {row['Memory_MB']} \\\\\n"
            
            latex_content += """\\hline
\\end{tabular}
\\end{table}
"""
        
        # Guardar archivo LaTeX
        with open(f"{output_dir}/tables_for_latex.tex", 'w') as f:
            f.write(latex_content)
        
        print(f"📄 Tabla LaTeX guardada: {output_dir}/tables_for_latex.tex")
    
    def generate_comprehensive_report(self, output_dir="comprehensive_report"):
        """Genera reporte comprehensivo completo"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("🚀 Generando reporte comprehensivo...")
        
        # Cargar todos los datos
        print("\n📊 Cargando métricas...")
        metrics_df = self.load_all_metrics()
        
        print("\n⏱️  Cargando timing...")
        timing_df = self.load_all_timing()
        
        if metrics_df is None:
            print("❌ No se pudieron cargar las métricas")
            return
        
        # Crear gráfico de violín para modelos 1024
        print("\n🎻 Creando gráfico de violín...")
        self.create_violin_plot_1024_targets(
            metrics_df, 
            os.path.join(output_dir, "violin_plot_1024_targets.png")
        )
        
        # Crear tablas de resumen
        print("\n📋 Creando tablas de resumen...")
        self.create_summary_tables(metrics_df, timing_df, 
                                  os.path.join(output_dir, "summary_tables"))
        
        # Crear análisis estadístico adicional
        print("\n📈 Creando análisis estadístico...")
        self._create_additional_analysis(metrics_df, timing_df, output_dir)
        
        print(f"\n🎉 Reporte completo generado en: {output_dir}")
    
    def _create_additional_analysis(self, metrics_df, timing_df, output_dir):
        """Crea análisis estadístico adicional"""
        
        # Análisis de correlaciones
        if metrics_df is not None:
            color_metrics = metrics_df[metrics_df['image_type'] == 'color']
            correlation_data = color_metrics[self.metrics + ['scale_factor']].corr()
            
            plt.figure(figsize=(12, 10))
            
            # Crear máscara triangular
            mask = np.triu(np.ones_like(correlation_data, dtype=bool))
            
            sns.heatmap(correlation_data, annot=True, cmap='RdBu_r', center=0,
                       square=True, fmt='.3f', mask=mask,
                       cbar_kws={"shrink": .8})
            
            plt.title('Correlation Matrix of Metrics and Scale Factor\n(Color Images Only)', 
                     fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Guardar matriz de correlación
            correlation_data.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))
            print("📊 Matriz de correlación guardada")
        
        # Análisis de eficiencia (si hay timing)
        if timing_df is not None and metrics_df is not None:
            self._create_efficiency_analysis(metrics_df, timing_df, output_dir)
    
    def _create_efficiency_analysis(self, metrics_df, timing_df, output_dir):
        """Crea análisis de eficiencia (calidad vs velocidad)"""
        
        # Combinar métricas y timing
        color_metrics = metrics_df[metrics_df['image_type'] == 'color']
        gpu_timing = timing_df[timing_df['device'] == 'gpu']
        
        # Calcular promedios por modelo
        metrics_avg = color_metrics.groupby('model_name')[self.metrics].mean().reset_index()
        timing_avg = gpu_timing.groupby('model_name')[['mean_time_ms', 'fps']].mean().reset_index()
        
        # Combinar datos
        efficiency_data = pd.merge(metrics_avg, timing_avg, on='model_name', how='inner')
        
        if len(efficiency_data) > 0:
            # Gráfico de eficiencia: PSNR vs Tiempo
            plt.figure(figsize=(12, 8))
            
            scatter = plt.scatter(efficiency_data['mean_time_ms'], efficiency_data['psnr'], 
                                s=100, alpha=0.7, c=efficiency_data['ssim'], 
                                cmap='viridis')
            
            for i, model in enumerate(efficiency_data['model_name']):
                plt.annotate(model, 
                           (efficiency_data['mean_time_ms'].iloc[i], efficiency_data['psnr'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points')
            
            plt.xlabel('Inference Time (ms)')
            plt.ylabel('PSNR (dB)')
            plt.title('Quality vs Speed Trade-off (GPU Inference)', fontweight='bold')
            plt.colorbar(scatter, label='SSIM')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "efficiency_analysis.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Guardar datos de eficiencia
            efficiency_data.to_csv(os.path.join(output_dir, "efficiency_data.csv"), index=False)
            print("📈 Análisis de eficiencia guardado")
        
        # Restaurar configuración por defecto
        plt.rcdefaults()
        
def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Análisis de Resultados ESRGAN")
    parser.add_argument("--validation_dir", default="Validation", 
                       help="Directorio con resultados de validación")
    parser.add_argument("--output_dir", default="comprehensive_report",
                       help="Directorio para guardar reporte")
    
    args = parser.parse_args()
    
    # Verificar que existe el directorio
    if not os.path.exists(args.validation_dir):
        print(f"❌ Error: Directorio no encontrado: {args.validation_dir}")
        return 1
    
    print("📊 SISTEMA DE ANÁLISIS DE RESULTADOS ESRGAN")
    print("=" * 50)
    print(f"Directorio de validación: {args.validation_dir}")
    print(f"Directorio de salida: {args.output_dir}")
    
    try:
        # Crear analizador
        analyzer = ESRGANResultsAnalyzer(args.validation_dir)
        
        # Generar reporte completo
        analyzer.generate_comprehensive_report(args.output_dir)
        
        print("\n✅ Análisis completado exitosamente!")
        
    except Exception as e:
        print(f"\n💥 Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())