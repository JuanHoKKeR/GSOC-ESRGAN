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
    
    def create_violin_plots_1024_targets(self, metrics_df, output_dir="violin_plots"):
        """
        Crea gráficos de violín individuales para modelos que van a resolución 1024x1024
        
        Args:
            metrics_df: DataFrame con todas las métricas
            output_dir: Directorio para guardar los gráficos individuales
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Filtrar modelos que van a 1024
        target_1024_models = ['64to1024', '128to1024', '256to1024', '512to1024']
        
        # Análisis tanto para color como escala de grises
        image_types = ['color', 'grayscale']
        
        for image_type in image_types:
            df_1024 = metrics_df[
                (metrics_df['base_model'].isin(target_1024_models)) & 
                (metrics_df['image_type'] == image_type)
            ].copy()
            
            if df_1024.empty:
                print(f"❌ No se encontraron datos {image_type} para modelos que van a 1024")
                continue
            
            print(f"📊 Creando gráficos de violín para imágenes {image_type}...")
            
            # Configurar estilo
            sns.set_style("whitegrid")
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            # Crear violín individual para cada métrica
            metrics_to_plot = ['psnr', 'ssim', 'ms_ssim', 'perceptual_index', 'mse']
            
            for metric in metrics_to_plot:
                if metric not in df_1024.columns:
                    continue
                
                # Crear figura individual
                plt.figure(figsize=(10, 6))
                
                # Crear el violin plot
                violin_parts = plt.violinplot(
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
                plt.xticks(range(len(target_1024_models)),
                          [f'{model}\n(×{self.model_configs[model]["scale"]})' 
                           for model in target_1024_models])
                
                # Título y labels
                metric_name = metric.upper().replace('_', '-')
                if metric == 'perceptual_index':
                    plt.ylabel(f'{metric_name} (lower is better)')
                elif metric == 'mse':
                    plt.ylabel(f'{metric_name} (lower is better)')
                else:
                    plt.ylabel(f'{metric_name} (higher is better)')
                
                plt.title(f'{metric_name} Distribution by Scale Factor ({image_type.title()} Images)', 
                         fontweight='bold', pad=20)
                plt.xlabel('Model (Scale Factor)')
                plt.grid(True, alpha=0.3)
                
                # Agregar estadísticas en el gráfico
                stats_text = []
                for j, model in enumerate(target_1024_models):
                    model_data = df_1024[df_1024['base_model'] == model][metric]
                    mean_val = model_data.mean()
                    std_val = model_data.std()
                    stats_text.append(f'{model}: μ={mean_val:.4f}±{std_val:.4f}')
                
                # Agregar texto de estadísticas
                stats_str = '\n'.join(stats_text)
                plt.text(0.02, 0.98, stats_str, transform=plt.gca().transAxes, 
                        verticalalignment='top', fontsize=9, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Guardar gráfico individual
                output_path = os.path.join(output_dir, f'violin_{metric}_{image_type}_1024targets.png')
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"  ✅ {metric.upper()} ({image_type}) guardado: {output_path}")
            
            # Crear tabla estadística para cada tipo de imagen
            self._create_statistics_table_1024(df_1024, target_1024_models, 
                                              os.path.join(output_dir, f'stats_1024targets_{image_type}.csv'))
    
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
        
        # Crear gráficos de violín individuales para modelos 1024 (MEJORADO)
        print("\n🎻 Creando gráficos de violín individuales...")
        violin_dir = os.path.join(output_dir, "violin_plots")
        self.create_violin_plots_1024_targets(metrics_df, violin_dir)
        
        # Crear tablas de resumen
        print("\n📋 Creando tablas de resumen...")
        self.create_summary_tables(metrics_df, timing_df, 
                                  os.path.join(output_dir, "summary_tables"))
        
        # Crear reporte estadístico simple (NUEVO)
        print("\n📄 Creando reporte estadístico simple...")
        self.create_simple_statistics_report(
            metrics_df, timing_df,
            os.path.join(output_dir, "statistics_summary.txt")
        )
        
        # Crear análisis estadístico adicional
        print("\n📈 Creando análisis estadístico...")
        self._create_additional_analysis(metrics_df, timing_df, output_dir)
        
        print(f"\n🎉 Reporte completo generado en: {output_dir}")
        print(f"\n📁 Estructura de archivos generados:")
        print(f"   📊 violin_plots/          - Gráficos individuales por métrica")
        print(f"   📋 summary_tables/        - Tablas de resumen y LaTeX")
        print(f"   📄 statistics_summary.txt - Reporte de texto simple")
        print(f"   📈 correlation_matrix.*   - Matriz de correlaciones")
        print(f"   ⚡ efficiency_analysis_*  - Análisis GPU y CPU")
    
    def _create_additional_analysis(self, metrics_df, timing_df, output_dir):
        """Crea análisis estadístico adicional"""
        
        # Análisis de correlaciones CON EXPLICACIÓN DETALLADA
        if metrics_df is not None:
            color_metrics = metrics_df[metrics_df['image_type'] == 'color']
            correlation_data = color_metrics[self.metrics + ['scale_factor']].corr()
            
            # Crear gráfico de correlaciones mejorado
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_data, dtype=bool))  # Máscara triangular
            
            sns.heatmap(correlation_data, annot=True, cmap='RdBu_r', center=0,
                       square=True, fmt='.3f', mask=mask,
                       cbar_kws={"shrink": .8, "label": "Correlation Coefficient"})
            
            plt.title('Correlation Matrix of Metrics and Scale Factor\n' +
                     '(Color Images Only)', fontweight='bold', fontsize=14, pad=20)
            plt.xlabel('Metrics')
            plt.ylabel('Metrics')
            
            # Agregar explicación en el gráfico
            explanation = ("Interpretation:\n"
                          "• +1: Perfect positive correlation\n"
                          "• 0: No linear correlation\n"
                          "• -1: Perfect negative correlation\n"
                          "• |r| > 0.7: Strong correlation\n"
                          "• |r| < 0.3: Weak correlation")
            
            plt.text(1.15, 0.5, explanation, transform=plt.gca().transAxes,
                    verticalalignment='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Guardar matriz de correlación con interpretación
            correlation_data.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))
            
            # Crear archivo explicativo de correlaciones
            with open(os.path.join(output_dir, "correlation_interpretation.txt"), 'w') as f:
                f.write("INTERPRETACIÓN DE LA MATRIZ DE CORRELACIÓN\n")
                f.write("=" * 50 + "\n\n")
                f.write("¿Qué mide la correlación?\n")
                f.write("La correlación mide la relación lineal entre dos variables.\n\n")
                f.write("Valores del coeficiente de correlación:\n")
                f.write("• +1.0: Correlación positiva perfecta (cuando una sube, la otra también)\n")
                f.write("• +0.7 a +0.9: Correlación positiva fuerte\n")
                f.write("• +0.3 a +0.7: Correlación positiva moderada\n")
                f.write("• -0.3 a +0.3: Correlación débil (casi independientes)\n")
                f.write("• -0.3 a -0.7: Correlación negativa moderada\n")
                f.write("• -0.7 a -0.9: Correlación negativa fuerte\n")
                f.write("• -1.0: Correlación negativa perfecta (cuando una sube, la otra baja)\n\n")
                
                f.write("Interpretación práctica para este estudio:\n")
                f.write("• PSNR vs SSIM: Si están altamente correlacionados, miden calidad similar\n")
                f.write("• MSE vs PSNR: Deberían tener correlación negativa (más error = menos PSNR)\n")
                f.write("• Perceptual_index vs otras métricas: Muestra si la percepción humana\n")
                f.write("  coincide con métricas matemáticas\n")
                f.write("• Scale_factor vs métricas: Muestra impacto del factor de escala\n\n")
                
                f.write("Hallazgos específicos en tus datos:\n")
                f.write("-" * 40 + "\n")
                
                # Análisis automático de correlaciones fuertes
                for i, metric1 in enumerate(correlation_data.columns):
                    for j, metric2 in enumerate(correlation_data.columns):
                        if i < j:  # Evitar duplicados
                            corr_val = correlation_data.iloc[i, j]
                            if abs(corr_val) > 0.7:
                                relation = "fuertemente correlacionados" if corr_val > 0 else "fuertemente anti-correlacionados"
                                f.write(f"• {metric1.upper()} y {metric2.upper()}: {relation} (r={corr_val:.3f})\n")
            
            print(f"📊 Matriz de correlación guardada con interpretación")
        
        # Análisis de eficiencia mejorado (si hay timing)
        if timing_df is not None and metrics_df is not None:
            self._create_efficiency_analysis(metrics_df, timing_df, output_dir)
    
    def _create_efficiency_analysis(self, metrics_df, timing_df, output_dir):
        """Crea análisis de eficiencia para GPU y CPU por separado"""
        
        # Combinar métricas y timing para GPU y CPU
        color_metrics = metrics_df[metrics_df['image_type'] == 'color']
        metrics_avg = color_metrics.groupby('model_name')[self.metrics].mean().reset_index()
        
        # Análisis para GPU y CPU por separado
        devices = ['gpu', 'cpu']
        
        for device in devices:
            device_timing = timing_df[timing_df['device'] == device]
            
            if len(device_timing) == 0:
                print(f"⚠️  No hay datos de timing para {device.upper()}")
                continue
                
            timing_avg = device_timing.groupby('model_name')[['mean_time_ms', 'fps']].mean().reset_index()
            
            # Combinar datos
            efficiency_data = pd.merge(metrics_avg, timing_avg, on='model_name', how='inner')
            
            if len(efficiency_data) == 0:
                print(f"⚠️  No hay datos combinados para {device.upper()}")
                continue
            
            # Gráfico de eficiencia: PSNR vs Tiempo
            plt.figure(figsize=(12, 8))
            
            scatter = plt.scatter(efficiency_data['mean_time_ms'], efficiency_data['psnr'], 
                                s=120, alpha=0.7, c=efficiency_data['ssim'], 
                                cmap='viridis', edgecolors='black', linewidth=0.5)
            
            for i, model in enumerate(efficiency_data['model_name']):
                plt.annotate(model.replace('_', '\n'), 
                           (efficiency_data['mean_time_ms'].iloc[i], efficiency_data['psnr'].iloc[i]),
                           xytext=(8, 8), textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            plt.xlabel('Inference Time (ms)')
            plt.ylabel('PSNR (dB)')
            plt.title(f'Quality vs Speed Trade-off ({device.upper()} Inference)', 
                     fontweight='bold', fontsize=14)
            
            # Colorbar con mejor formato
            cbar = plt.colorbar(scatter, label='SSIM')
            cbar.ax.tick_params(labelsize=10)
            
            plt.grid(True, alpha=0.3)
            
            # Agregar información en el gráfico
            info_text = f"Device: {device.upper()}\nModels: {len(efficiency_data)}\n"
            info_text += f"Time range: {efficiency_data['mean_time_ms'].min():.1f}-{efficiency_data['mean_time_ms'].max():.1f} ms\n"
            info_text += f"PSNR range: {efficiency_data['psnr'].min():.2f}-{efficiency_data['psnr'].max():.2f} dB"
            
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"efficiency_analysis_{device}.png"), dpi=300)
            plt.close()
            
            # Guardar datos de eficiencia por dispositivo
            efficiency_data.to_csv(os.path.join(output_dir, f"efficiency_data_{device}.csv"), index=False)
            
            print(f"📈 Análisis de eficiencia {device.upper()} guardado")
    
    def create_simple_statistics_report(self, metrics_df, timing_df, output_path="statistics_summary.txt"):
        """
        Crea archivo de texto simple con estadísticas resumidas para análisis rápido
        
        Args:
            metrics_df: DataFrame con métricas
            timing_df: DataFrame con timing
            output_path: Ruta del archivo de texto
        """
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RESUMEN ESTADÍSTICO RÁPIDO - MODELOS ESRGAN\n")
            f.write("=" * 80 + "\n\n")
            
            # Análisis por modelo y tipo de imagen
            for image_type in ['color', 'grayscale']:
                f.write(f"\n{'='*60}\n")
                f.write(f"MÉTRICAS - IMÁGENES {image_type.upper()}\n")
                f.write(f"{'='*60}\n\n")
                
                type_data = metrics_df[metrics_df['image_type'] == image_type]
                
                for model_name in sorted(type_data['model_name'].unique()):
                    model_data = type_data[type_data['model_name'] == model_name]
                    base_model = model_data['base_model'].iloc[0]
                    
                    f.write(f"Modelo: {model_name}\n")
                    f.write(f"Configuración base: {base_model}\n")
                    f.write(f"Factor de escala: ×{self.model_configs.get(base_model, {}).get('scale', '?')}\n")
                    f.write(f"Muestras analizadas: {len(model_data)}\n")
                    
                    # Parámetros arquitectónicos
                    params = self.model_configs.get(base_model, {}).get('params', {})
                    f.write(f"Arquitectura: NF={params.get('num_features', '?')}, ")
                    f.write(f"TS={params.get('trunk_size', '?')}, ")
                    f.write(f"GC={params.get('growth_channel', '?')}\n")
                    
                    if params.get('num_features', 32) != 32:
                        f.write("*** MODELO OPTIMIZADO (parámetros modificados) ***\n")
                    
                    f.write("\nMétricas (Media ± Desv. Estándar):\n")
                    f.write("-" * 40 + "\n")
                    
                    for metric in self.metrics:
                        if metric in model_data.columns:
                            mean_val = model_data[metric].mean()
                            std_val = model_data[metric].std()
                            min_val = model_data[metric].min()
                            max_val = model_data[metric].max()
                            
                            f.write(f"  {metric.upper():15}: {mean_val:8.4f} ± {std_val:6.4f} ")
                            f.write(f"(min: {min_val:7.4f}, max: {max_val:7.4f})\n")
                    
                    f.write("\n" + "-" * 70 + "\n\n")
            
            # Análisis de timing
            if timing_df is not None:
                f.write(f"\n{'='*60}\n")
                f.write("ANÁLISIS DE TIMING (VELOCIDAD DE INFERENCIA)\n")
                f.write(f"{'='*60}\n\n")
                
                for device in ['gpu', 'cpu']:
                    f.write(f"\n--- {device.upper()} ---\n")
                    device_data = timing_df[timing_df['device'] == device]
                    
                    if len(device_data) == 0:
                        f.write(f"No hay datos de timing para {device.upper()}\n\n")
                        continue
                    
                    for model_name in sorted(device_data['model_name'].unique()):
                        model_timing = device_data[device_data['model_name'] == model_name]
                        
                        f.write(f"\nModelo: {model_name}\n")
                        f.write(f"Muestras: {len(model_timing)}\n")
                        
                        # Estadísticas de timing
                        time_mean = model_timing['mean_time_ms'].mean()
                        time_std = model_timing['mean_time_ms'].std()
                        fps_mean = model_timing['fps'].mean()
                        memory_mean = model_timing['memory_increase_mb'].mean()
                        
                        f.write(f"Tiempo promedio:     {time_mean:8.2f} ± {time_std:6.2f} ms\n")
                        f.write(f"FPS promedio:        {fps_mean:8.2f}\n")
                        f.write(f"Memoria adicional:   {memory_mean:8.1f} MB\n")
                        f.write("-" * 50 + "\n")
            
            # Comparación Color vs Escala de Grises
            f.write(f"\n{'='*60}\n")
            f.write("COMPARACIÓN: COLOR vs ESCALA DE GRISES\n")
            f.write(f"{'='*60}\n\n")
            
            for model_name in sorted(metrics_df['model_name'].unique()):
                color_data = metrics_df[(metrics_df['model_name'] == model_name) & 
                                      (metrics_df['image_type'] == 'color')]
                gray_data = metrics_df[(metrics_df['model_name'] == model_name) & 
                                     (metrics_df['image_type'] == 'grayscale')]
                
                if len(color_data) == 0 or len(gray_data) == 0:
                    continue
                
                f.write(f"\nModelo: {model_name}\n")
                f.write(f"{'Métrica':<15} {'Color':<12} {'Escala Grises':<15} {'Diferencia':<10}\n")
                f.write("-" * 60 + "\n")
                
                for metric in self.metrics:
                    if metric in color_data.columns and metric in gray_data.columns:
                        color_mean = color_data[metric].mean()
                        gray_mean = gray_data[metric].mean()
                        diff = color_mean - gray_mean
                        
                        f.write(f"{metric.upper():<15} {color_mean:<12.4f} {gray_mean:<15.4f} {diff:<10.4f}\n")
                
                f.write("\n")
            
            # Resumen de hallazgos
            f.write(f"\n{'='*60}\n")
            f.write("HALLAZGOS PRINCIPALES\n")
            f.write(f"{'='*60}\n\n")
            
            # Encontrar mejores y peores modelos por métrica
            color_metrics = metrics_df[metrics_df['image_type'] == 'color']
            model_averages = color_metrics.groupby('model_name')[self.metrics].mean()
            
            f.write("Mejores modelos por métrica (imágenes color):\n")
            f.write("-" * 45 + "\n")
            
            for metric in self.metrics:
                if metric in model_averages.columns:
                    if metric in ['perceptual_index', 'mse']:
                        best_model = model_averages[metric].idxmin()
                        best_value = model_averages[metric].min()
                        f.write(f"{metric.upper():<15}: {best_model:<20} ({best_value:.4f})\n")
                    else:
                        best_model = model_averages[metric].idxmax()
                        best_value = model_averages[metric].max()
                        f.write(f"{metric.upper():<15}: {best_model:<20} ({best_value:.4f})\n")
            
            f.write("\nImpacto de modelos optimizados:\n")
            f.write("-" * 35 + "\n")
            optimized_models = [model for model in model_averages.index 
                              if 'Optimized' in model or 
                              self.model_configs.get(self._extract_base_model(model), {})
                              .get('params', {}).get('num_features', 32) != 32]
            
            if optimized_models:
                f.write("Modelos con arquitectura optimizada/modificada:\n")
                for model in optimized_models:
                    base_model = self._extract_base_model(model)
                    params = self.model_configs.get(base_model, {}).get('params', {})
                    f.write(f"  - {model}: NF={params.get('num_features', '?')}, ")
                    f.write(f"TS={params.get('trunk_size', '?')}, ")
                    f.write(f"GC={params.get('growth_channel', '?')}\n")
            else:
                f.write("No se identificaron modelos optimizados en los datos.\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("Fin del reporte\n")
            f.write(f"{'='*80}\n")
        
        print(f"📄 Reporte estadístico simple guardado: {output_path}")

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