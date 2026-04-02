#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
bool isprime[2000005];
int prime[2000005];
int cnt = 0;
void euler(int n)
{
    isprime[0] = isprime[1] = 1;
    for (int i = 2; i <= n; i++)
    {
        if (!isprime[i])
            prime[++cnt] = i;
        for (int j = 1; j <= cnt && i * prime[j] <= n; j++)
        {
            isprime[i * prime[j]] = 1;
            if (i % prime[j] == 0)
                break;
        }
    }
}
void solve()
{
    int n, k;
    cin >> n >> k;
    int t = n;
    bool ok = 1;
    int cntt = 0;
    while (t > 0)
    {
        int x = t % 10;
        if (x != 1)
        {
            ok = 0;
            break;
        }
        t /= 10;
        cntt++;
    }
    if (ok)
    {
        cntt *= k;
        if (cntt == 2 || cntt == 19 || cntt == 23)
            cout << "YES\n";
        else
            cout << "NO\n";
        return;
    }
    if (k == 1)
    {
        for (int i = 2; i * i <= n; i++)
        {
            if (n % i == 0)
            {
                cout << "NO\n";
                return;
            }
        }
        cout << "YES\n";
    }
    else
    {
        cout << "NO\n";
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    euler(90);
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}