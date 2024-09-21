import axios from 'axios'

export async function POST(request) {
	const formData = await request.formData()
	const file = formData.get('file')

	if (!file) {
		return new Response(JSON.stringify({ error: 'No file uploaded' }), {
			status: 400,
		})
	}

	try {
		const response = await axios.post(
			'http://localhost:5000/predict',
			formData,
			{
				headers: { 'Content-Type': 'multipart/form-data' },
			}
		)

		return new Response(JSON.stringify(response.data), { status: 200 })
	} catch (error) {
		return new Response(JSON.stringify({ error: error.message }), {
			status: 500,
		})
	}
}
