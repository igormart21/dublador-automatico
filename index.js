require('dotenv').config();
const express = require('express');
const axios = require('axios');

const app = express();

// Para conseguir ler JSON no body das requisições
app.use(express.json());

// Configuração da API Hotmart
const HOTMART_API_URL = 'https://api-sec-vlc.hotmart.com';
const HOTMART_AUTH_URL = 'https://api-sec-vlc.hotmart.com/security/oauth/token';
const HOTMART_CLUB_URL = 'https://api-sec-vlc.hotmart.com/club/api/v1';

// Função para obter token de acesso da Hotmart
async function getHotmartToken() {
  try {
    console.log('Iniciando requisição de token para Hotmart...');
    const auth = Buffer.from(`${process.env.HOTMART_CLIENT_ID}:${process.env.HOTMART_CLIENT_SECRET}`).toString('base64');
    
    console.log('Usando credenciais:', {
      clientId: process.env.HOTMART_CLIENT_ID,
      clientSecret: process.env.HOTMART_CLIENT_SECRET,
      auth: auth.substring(0, 20) + '...'
    });

    const response = await axios.post(HOTMART_AUTH_URL, 'grant_type=client_credentials', {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': `Basic ${auth}`
      }
    });

    console.log('Resposta completa da autenticação Hotmart:', response);
    return response.data.access_token;
  } catch (error) {
    console.error('Erro ao obter token:', error.response?.data || error.message);
    throw error;
  }
}

// Função para registrar usuário na Hotmart
async function registerHotmartUser(userData) {
  try {
    const token = await getHotmartToken();
    console.log('Token obtido com sucesso:', token.substring(0, 20) + '...');
    
    console.log('Enviando requisição para Hotmart com os dados:', userData);

    // Tentar diferentes endpoints da API
    const endpoints = [
      `${HOTMART_CLUB_URL}/members`,
      `${HOTMART_CLUB_URL}/memberships`,
      `${HOTMART_API_URL}/club/api/v2/memberships`,
      `${HOTMART_API_URL}/club/api/v2/subscription`,
      `${HOTMART_CLUB_URL}/users/register`
    ];

    for (const endpoint of endpoints) {
      try {
        console.log('Tentando endpoint:', endpoint);
        const response = await axios.post(endpoint, userData, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        });
        console.log('Resposta do endpoint:', endpoint, response.data);
        return response.data;
      } catch (error) {
        console.log('Erro no endpoint:', endpoint, error.response?.status);
        continue;
      }
    }

    throw new Error('Todos os endpoints falharam');
  } catch (error) {
    console.error('Erro detalhado na comunicação com a Hotmart:', error.response || error);
    throw new Error('Falha na integração com a Hotmart: ' + error.message);
  }
}

// Rota que a CartPanda chamará (Webhook)
app.post('/webhook-cartpanda', async (req, res) => {
  try {
    console.log('Recebendo webhook da CartPanda:', req.body);

    // 1) Extrair e validar dados enviados pela CartPanda
    const {
      name,
      email,
      productId = process.env.DEFAULT_PRODUCT_ID,
      transactionStatus,
      phone,
      transaction_id
    } = req.body;

    // Validações básicas
    if (!email) {
      console.log('Email não fornecido');
      return res.status(400).json({ error: 'E-mail é obrigatório' });
    }

    if (!transactionStatus) {
      console.log('Status da transação não fornecido');
      return res.status(400).json({ error: 'Status da transação é obrigatório' });
    }

    // 2) Verificar se o pagamento foi APROVADO
    if (transactionStatus === 'approved') {
      console.log('Aguardando webhook da Hotmart para o pedido:', transaction_id);
      
      // Retornar sucesso para a CartPanda
      // A Hotmart enviará o webhook quando o pagamento for processado
      return res.status(200).json({ 
        success: true,
        message: 'Pedido registrado com sucesso. Aguardando confirmação da Hotmart.'
      });
    } else {
      console.log(`Transação não aprovada. Status: ${transactionStatus}`);
    }

    // 3) Retornar resposta de sucesso à CartPanda
    return res.status(200).json({ 
      success: true,
      message: transactionStatus === 'approved' ? 'Pedido registrado com sucesso' : 'Status registrado com sucesso'
    });

  } catch (err) {
    console.error('Erro na integração CartPanda -> Hotmart:', err.message);
    return res.status(500).json({ 
      error: 'Erro ao processar webhook.',
      message: err.message 
    });
  }
});

// Rota para receber webhooks da Hotmart
app.post('/webhook-hotmart', async (req, res) => {
  try {
    console.log('Recebendo webhook da Hotmart:', req.body);

    // Extrair dados do webhook
    const { event, data } = req.body;
    const { purchase, buyer, product } = data || {};

    console.log('Evento recebido:', event);
    console.log('Dados da compra:', {
      transaction: purchase?.transaction,
      status: purchase?.status,
      buyer: {
        name: buyer?.name,
        email: buyer?.email,
        phone: buyer?.checkout_phone
      },
      product: {
        id: product?.id,
        name: product?.name
      }
    });

    // Verificar se é uma compra aprovada
    if (event === 'PURCHASE_APPROVED' && purchase?.status === 'APPROVED') {
      console.log('Compra aprovada, processando...');
      
      // Aqui você pode adicionar a lógica para:
      // 1. Registrar o usuário no seu sistema
      // 2. Enviar email de boas-vindas
      // 3. Criar conta na área de membros
      
      console.log('Compra processada com sucesso');
    } else {
      console.log('Evento não é PURCHASE_APPROVED ou status não é APPROVED:', { event, status: purchase?.status });
    }

    // Retornar sucesso para a Hotmart
    return res.status(200).json({ 
      success: true,
      message: 'Webhook processado com sucesso'
    });

  } catch (err) {
    console.error('Erro ao processar webhook da Hotmart:', err);
    return res.status(500).json({ 
      error: 'Erro ao processar webhook.',
      message: err.message 
    });
  }
});

// Rota de healthcheck
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok' });
});

// Iniciar servidor
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Servidor rodando na porta ${PORT}`);
  console.log(`Healthcheck disponível em: http://localhost:${PORT}/health`);
});
