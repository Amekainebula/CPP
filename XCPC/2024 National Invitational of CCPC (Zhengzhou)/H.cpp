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
int p = 998244353;
int gcd(int a, int b)
{
    return b ? gcd(b, a % b) : a;
}

int lcm(int a, int b)
{
    return a / gcd(a, b) * b;
}

int Pow(int a, int b)
{ // a ^ b
    int ans = 1;
    while (b)
    {
        if (b & 1)
        {
            ans = ans * a;
        }
        a *= a;
        b >>= 1;
    }
    return ans;
}

int ModPow(int a, int b)
{ // a ^ b % p
    int ans = 1;
    while (b)
    {
        if (b & 1)
        {
            ans = (ans * a) % p;
        }
        a = (a * a) % p;
        b >>= 1;
    }
    return ans;
}

int inv(int a)
{
    return ModPow(a, p - 2);
}
void Murasame()
{
    int n;
    cin >> n;
    vi a, test(2 * n + 1);
    ff(i, 1, 2 * n)
    {
        cin >> test[i];
        if (test[i] != -1)
            a.eb(test[i]);
    }
    sort(all(a));
    int now = 0;
    int has = 0;
    map<int, int> mp;
    int fz = 1;
    int fm = 1;
    ff(i, 1, 2 * n)
    {
        if (test[i] != -1)
        {
            mp[test[i]]++;
            has++;
        }
        else
        {
            fz *= mp[a[now]];
            fz %= p;
            fm *= has;
            fm %= p;
            has--;
            mp[a[now]]--;
            now++;
        }
        if (fz == 0)
        {
            cout << "0\n";
            return;
        }
    }
    cout << fz * inv(fm) % p << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}