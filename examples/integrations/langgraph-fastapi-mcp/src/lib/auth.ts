import { betterAuth } from "better-auth";
import { drizzleAdapter } from "better-auth/adapters/drizzle";
import { db } from "../../db/drizzle"; // your drizzle instance
import { schema } from "../../db/schema";
import { nextCookies } from "better-auth/next-js";

export const auth = betterAuth({
  loginPath: "/pages/login", // The path where your auth API routes are located
  emailAndPassword: {
    enabled: true,
  },
  database: drizzleAdapter(db, {
    provider: "pg", // or "mysql", "sqlite"
    schema: schema,
  }),

  socialProviders: {
    cognito: {
      clientId: process.env.COGNITO_CLIENT_ID as string,
      clientSecret: process.env.COGNITO_CLIENT_SECRET as string,
      domain: process.env.COGNITO_DOMAIN as string, // e.g. "your-app.auth.us-east-1.amazoncognito.com"
      region: process.env.COGNITO_REGION as string, // e.g. "us-east-1"
      userPoolId: process.env.COGNITO_USER_POOL_ID as string,
    },
  },
  plugins: [nextCookies()],
});
