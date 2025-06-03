document.addEventListener('DOMContentLoaded', () => {
    // Create cursor elements
    const cursorDot = document.createElement('div');
    const cursorCircle = document.createElement('div');
    
    cursorDot.className = 'cursor-dot';
    cursorCircle.className = 'cursor-circle';
    
    document.body.appendChild(cursorDot);
    document.body.appendChild(cursorCircle);
    
    // Update cursor position
    document.addEventListener('mousemove', (e) => {
        requestAnimationFrame(() => {
            cursorDot.style.left = e.clientX + 'px';
            cursorDot.style.top = e.clientY + 'px';
            
            cursorCircle.style.left = e.clientX + 'px';
            cursorCircle.style.top = e.clientY + 'px';
        });
    });
});