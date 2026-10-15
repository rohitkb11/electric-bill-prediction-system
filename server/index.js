import { app } from './app.js';
const port = process.env.PORT || 3000;
app.listen(port, '127.0.0.1', () => console.log(`Electric Bill Prediction System: http://localhost:${port}`));
