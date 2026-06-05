"use server";

import { tr } from "zod/v4/locales";
import { auth } from "../src/lib/auth";
import { success } from "zod";

export const signIn = async (email: string, password: string) => {
  try {
    await auth.api.signInEmail({
      body: {
        email: email,
        password: password,
      },
    });

    return {
      success: true,
      message: "User signed in successfully.",
    };
  } catch (error) {
    const e = error as Error;
    return {
      success: false,
      message: e.message || "An error occurred during sign-in.",
    };
  }
};

export const signUp = async (
  username: string,
  email: string,
  password: string,
) => {
  try {
    await auth.api.signUpEmail({
      body: {
        email: email,
        password: password,
        name: username,
      },
    });

    return {
      success: true,
      message: "User signed up successfully.",
    };
  } catch (error) {
    const e = error as Error;
    return {
      success: false,
      message: e.message || "An error occurred during sign-in.",
    };
  }
};
