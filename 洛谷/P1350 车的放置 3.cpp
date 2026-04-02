#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
// const int MOD = 1e5 + 3;
const int mod = 1e5 + 3;
const int MOD = 998244353;
const int N = 4e6 + 5;
int qpow(int a, int b)
{
    int res = 1;
    while (b)
    {
        if (b & 1)
            res = res * a % MOD;
        a = a * a % MOD, b >>= 1;
    }
    return res;
}
int getinv(int x) { return qpow(x, MOD - 2); }
int fact[N], inv_fact[N];
int inv[N];
void init_comb(int n)
{
    fact[0] = 1;
    for (int i = 1; i <= n; i++)
        fact[i] = fact[i - 1] * i % MOD;
    inv_fact[n] = getinv(fact[n]);
    for (int i = n; i >= 1; i--)
        inv_fact[i - 1] = inv_fact[i] * i % MOD;
    inv[1] = 1;
    for (int i = 2; i <= n; i++)
        inv[i] = (MOD - MOD / i) * inv[MOD % i] % MOD;
    return;
}
int comb(int n, int m)
{
    if (m > n)
        return 0;
    else if (m == 0 || m == n)
        return 1;
    else
        return fact[n] * inv_fact[m] % MOD * inv_fact[n - m] % MOD;
}
int dp[2003][2003];
void Murasame()
{
    int a, b, c, d, k;
    cin >> a >> b >> c >> d >> k;
    ff(i, 0, b + d) dp[i][0] = 1;
    ff(i, 1, b + d) ff(j, 1, k)
    {
        if (i <= b && j > a)
            break;
        dp[i][j] = (dp[i - 1][j] + dp[i - 1][j - 1] * (a - j + 1 + (i > b ? c : 0))) % mod;
    }
    cout << dp[b + d][k] << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // init_comb(N - 1);
    //  cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}