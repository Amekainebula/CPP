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
#define vvi vector<vi>
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
const int MOD = 1e9 + 7;
const int mod = 998244353;
int n;
int gcd(int a, int b) { return b ? gcd(b, a % b) : a; }
int N = 500005;
vector<int> in(N + 1);
vector<vector<int>> pre(N + 1);
vector<int> primes;
void euler(int N)
{
    in[1] = 1;
    for (int i = 2; i <= N; ++i)
    {
        if (!in[i])
        {
            in[i] = i;
            primes.push_back(i);
        }
        for (int p : primes)
        {
            int ip = 1LL * i * p;
            if (p > in[i] || ip > N)
                break;
            in[ip] = p;
        }
    }
    pre[1] = {1};
    for (int i = 2; i <= N; ++i)
    {
        int p = in[i], m = i / p, cnt = 1;
        while (m % p == 0)
        {
            m /= p;
            ++cnt;
        }
        for (int d : pre[m])
        {
            int cur = 1;
            for (int k = 0; k <= cnt; ++k)
            {
                pre[i].pb(d * cur);
                cur *= p;
            }
        }
    }
}
vi c(500005, 0);
void Murasame()
{
    fill(c.begin(), c.end(), 0);
    cin >> n;
    int ans = 0;
    vi a(n + 1);
    int maxx = 0;
    ff(i, 1, n) cin >> a[i], maxx = max(maxx, a[i]), c[a[i]]++;
    for (int i = 1; i <= maxx; ++i)
    {
        for (int j = 1; i * (j + 1) <= maxx; j++)
        {
            ans += c[i * (j + 1)] * c[i * j];
        }
    }
    cout << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    // euler(N);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}