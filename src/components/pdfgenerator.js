import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

export const generatePDF = (elementId, filename = 'analysis_report.pdf') => {
  const input = document.getElementById(elementId);
  html2canvas(input).then((canvas) => {
    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF('p', 'mm', 'a4');
    const width = pdf.internal.pageSize.getWidth();
    const height = (canvas.height * width) / canvas.width;
    pdf.addImage(imgData, 'PNG', 0, 0, width, height);
    pdf.save(filename);
  });
};